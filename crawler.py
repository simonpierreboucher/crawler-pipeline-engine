#!/usr/bin/env python3
# crawler.py

import os
import json
import requests
from pathlib import Path
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import cycle
from collections import defaultdict, deque
from urllib.parse import urljoin, urlparse
from typing import Optional, Set, Dict, List
import hashlib
import re
import html2text
import datetime
from bs4 import BeautifulSoup
from xml.etree.ElementTree import Element, SubElement, ElementTree
import threading

from pydantic import BaseModel, ValidationError
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, SpinnerColumn
from rich.logging import RichHandler
from rich.console import Console

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

CONTENT_REWRITER_OUTPUT_DIR = "./output/content_rewritten"

metrics_lock = threading.Lock()

metrics = {
    "urls_extracted": 0,
    "pages_extracted": 0,
    "pages_rewritten": 0,
    "files_downloaded": defaultdict(int),
    "errors": []
}


class ContentRewriter:
    def __init__(
        self, 
        input_dir: str, 
        output_dir: str, 
        api_keys: List[str], 
        system_prompt: str, 
        model: str = "openai", 
        verbose: bool = False, 
        max_tokens: int = 2048, 
        concurrency: int = 5
    ):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.api_keys = api_keys if api_keys else []
        if not self.api_keys:
            raise ValueError("Au moins une clé API est requise pour la réécriture du contenu.")
        self.api_keys_cycle = cycle(self.api_keys)
        self.lock = threading.Lock()
        
        self.model = model.lower()
        self.verbose = verbose
        self.max_tokens = max_tokens
        self.concurrency = concurrency
        
        self.system_prompt = system_prompt
        
        logging.basicConfig(
            level=logging.INFO if not verbose else logging.DEBUG,
            format='%(message)s',
            datefmt="[%X]",
            handlers=[RichHandler()]
        )
        self.logger = logging.getLogger(__name__)

        self.console = Console()

    def get_next_api_key(self) -> Optional[str]:
        with self.lock:
            try:
                return next(self.api_keys_cycle)
            except StopIteration:
                return None

    def split_text_into_chunks(self, text: str) -> List[str]:
        avg_chars_per_token = 4
        max_chars = self.max_tokens * avg_chars_per_token
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ""

        for para in paragraphs:
            if len(current_chunk) + len(para) + 2 <= max_chars:
                current_chunk += para + '\n\n'
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = para + '\n\n'
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        self.logger.debug(f"Texte divisé en {len(chunks)} morceaux.")
        return chunks

    def rewrite_chunk(self, chunk: str, document_name: str, file_name: str) -> Optional[str]:
        system_prompt = self.system_prompt
        user_content = chunk

        api_key = self.get_next_api_key()
        if not api_key:
            self.logger.error("Aucune clé API disponible pour la réécriture.")
            self._update_metrics('errors', "Aucune clé API disponible pour la réécriture.")
            return None

        if self.model == "openai":
            endpoint = "https://api.openai.com/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                "temperature": 0,
                "max_tokens": self.max_tokens,
                "top_p": 1
            }
        elif self.model == "anthropic":
            endpoint = "https://api.anthropic.com/v1/complete"
            headers = {
                "x-api-key": api_key,
                "Content-Type": "application/json"
            }
            payload = {
                "prompt": f"{system_prompt}\n\n{user_content}",
                "model": "claude-3.5",
                "max_tokens_to_sample": self.max_tokens,
                "temperature": 0,
                "top_p": 0
            }
        elif self.model == "mistral":
            endpoint = "https://api.mistral.ai/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json"
            }
            payload = {
                "model": "mistral-large-latest",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                "temperature": 0.7,
                "top_p": 1,
                "max_tokens": self.max_tokens,
                "stream": False
            }
        else:
            self.logger.error(f"Fournisseur LLM inconnu : {self.model}")
            self._update_metrics('errors', f"Fournisseur LLM inconnu : {self.model}")
            return None

        if self.verbose:
            self.logger.debug(f"Appel du LLM {self.model} pour {document_name} : Taille du payload {len(user_content)}")

        try:
            response = requests.post(endpoint, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
        except requests.HTTPError as e:
            self.logger.error(f"Erreur API LLM {self.model} : {str(e)}, réponse : {response.text}")
            self._update_metrics('errors', f"Erreur API LLM {self.model} : {str(e)}, réponse : {response.text}")
            return None
        except Exception as e:
            self.logger.error(f"Erreur lors de l'appel au LLM {self.model} : {str(e)}")
            self._update_metrics('errors', f"Erreur lors de l'appel au LLM {self.model} : {str(e)}")
            return None

        response_json = response.json()
        if self.model == "openai":
            rewritten_text = response_json['choices'][0]['message']['content']
        elif self.model == "anthropic":
            rewritten_text = response_json.get("completion", "")
        elif self.model == "mistral":
            rewritten_text = response_json['choices'][0]['message']['content']
        else:
            rewritten_text = ""

        return rewritten_text

    def rewrite_file(self, file_path: Path) -> bool:
        document_name = file_path.stem
        self.logger.info(f"Réécriture du contenu de {file_path.name}")

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        except Exception as e:
            self.logger.error(f"Erreur lors de la lecture de {file_path.name} : {str(e)}")
            self._update_metrics('errors', f"Erreur lors de la lecture de {file_path.name} : {str(e)}")
            return False

        if not text.strip():
            self.logger.warning(f"Aucun texte à réécrire dans {file_path.name}")
            self._update_metrics('errors', f"Aucun texte à réécrire dans {file_path.name}")
            return False

        chunks = self.split_text_into_chunks(text)
        total_chunks = len(chunks)
        rewritten_chunks = [None] * total_chunks

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeRemainingColumn(),
            transient=True,
            console=self.console
        ) as progress:
            task = progress.add_task(f"Réécriture de {file_path.name}", total=total_chunks)

            def process_chunk(index, chunk):
                rewritten = self.rewrite_chunk(chunk, document_name, file_path.name)
                rewritten_chunks[index] = rewritten
                progress.advance(task)

            with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
                futures = {executor.submit(process_chunk, idx, chunk): idx for idx, chunk in enumerate(chunks)}
                for future in as_completed(futures):
                    idx = futures[future]
                    try:
                        future.result()
                    except Exception as e:
                        self.logger.error(f"Erreur lors de la réécriture du segment {idx + 1} : {str(e)}")
                        self._update_metrics('errors', f"Erreur lors de la réécriture du segment {idx + 1} : {str(e)}")

        if any(chunk is None for chunk in rewritten_chunks):
            self.logger.warning(f"Réécriture partielle échouée pour {file_path.name}")
            return False

        rewritten_text = "\n\n".join(rewritten_chunks)
        output_file_name = self.output_dir / f"{document_name}_rewritten.txt"
        try:
            with open(output_file_name, 'w', encoding='utf-8') as f:
                f.write(rewritten_text)
            self.logger.info(f"Fichier réécrit créé : {output_file_name}")
            self._update_metrics('pages_rewritten', 1)
            return True
        except Exception as e:
            self.logger.error(f"Erreur lors de la sauvegarde de {file_path} : {str(e)}")
            self._update_metrics('errors', f"Erreur lors de la sauvegarde de {file_path} : {str(e)}")
            return False

    def rewrite_all_contents(self):
        txt_files = list(self.input_dir.glob('*.txt'))
        total_files = len(txt_files)
        self.logger.info(f"Démarrage de la réécriture de {total_files} fichier(s)")

        if total_files == 0:
            self.logger.info("Aucun fichier à réécrire.")
            return

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeRemainingColumn(),
            console=self.console
        ) as progress:
            task = progress.add_task("Réécriture des fichiers", total=total_files)

            with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
                futures = {executor.submit(self.rewrite_file, txt_file_path): txt_file_path for txt_file_path in txt_files}

                for future in as_completed(futures):
                    txt_file_path = futures[future]
                    try:
                        result = future.result()
                        if result:
                            self.logger.info(f"Réécriture réussie pour {txt_file_path.name}")
                        progress.advance(task)
                    except Exception as e:
                        self.logger.error(f"Erreur lors de la réécriture de {txt_file_path.name} : {str(e)}")
                        self._update_metrics('errors', f"Erreur lors de la réécriture de {txt_file_path.name} : {str(e)}")
                        progress.advance(task)
        self.logger.info("Réécriture du contenu terminée.")

    def _update_metrics(self, key: str, value):
        with metrics_lock:
            if key == 'errors':
                metrics['errors'].append(value)
            elif key == 'files_downloaded' and isinstance(value, str):
                metrics['files_downloaded'][value] += 1
            elif key in metrics:
                metrics[key] += value
            else:
                metrics[key] = value


class DownloadConfig(BaseModel):
    pdf: bool = False
    doc: bool = False
    image: bool = False
    other: bool = False


class OutputConfig(BaseModel):
    crawler_output_dir: str
    checkpoint_file: str
    logs_dir: str
    content_dir: str
    content_rewritten_dir: str
    PDF: str
    Image: str
    Doc: str
    Archive: str
    Audio: str
    Video: str


class LimitsConfig(BaseModel):
    max_tokens: int = 2048
    concurrency: int = 5
    max_urls: Optional[int] = None


class CrawlerConfig(BaseModel):
    start_url: str
    max_depth: int = 2
    use_playwright: bool = False
    download: DownloadConfig
    output: OutputConfig
    limits: LimitsConfig
    verbose: bool = False


class LLMConfig(BaseModel):
    provider: str = "openai"
    api_keys: List[str]
    max_tokens: int = 2048
    concurrency: int = 5
    system_prompt_file: str


class Config(BaseModel):
    crawler: CrawlerConfig
    llm: LLMConfig


class WebCrawler:
    def __init__(self, config: dict):
        try:
            self.config = CrawlerConfig.parse_obj(config['crawler'])
            self.llm_config = LLMConfig.parse_obj(config['llm'])
        except ValidationError as e:
            print("Erreur de validation de la configuration :", e)
            raise

        crawler_conf = self.config
        llm_conf = self.llm_config

        self.start_url = crawler_conf.start_url
        self.max_depth = crawler_conf.max_depth
        self.use_playwright = crawler_conf.use_playwright
        self.download_pdf = crawler_conf.download.pdf
        self.download_doc = crawler_conf.download.doc
        self.download_image = crawler_conf.download.image
        self.download_other = crawler_conf.download.other
        self.llm_provider = llm_conf.provider
        self.api_keys = llm_conf.api_keys
        self.llm_enabled = bool(self.llm_provider and self.api_keys)
        self.max_tokens_per_request = crawler_conf.limits.max_tokens
        self.concurrency = crawler_conf.limits.concurrency
        self.max_urls = crawler_conf.limits.max_urls
        self.base_dir = Path(crawler_conf.output.crawler_output_dir)
        self.checkpoint_file = Path(crawler_conf.output.checkpoint_file)

        self.visited_pages = set()
        self.downloaded_files = set()
        self.domain = urlparse(self.start_url).netloc
        self.site_map: Dict[str, Set[str]] = defaultdict(set)

        self.excluded_paths = ['selecteur-de-produits']

        (self.base_dir / crawler_conf.output.logs_dir).mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            level=logging.DEBUG if crawler_conf.verbose else logging.INFO,
            format='%(message)s',
            datefmt="[%X]",
            handlers=[RichHandler()]
        )
        self.logger = logging.getLogger(__name__)

        self.console = Console()

        self.stats = defaultdict(int)
        self.downloadable_extensions = {
            'PDF': ['.pdf'] if self.download_pdf else [],
            'Image': ['.png', '.jpg', '.jpeg', '.gif', '.svg'] if self.download_image else [],
            'Doc': ['.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx'] if self.download_doc else [],
            'Archive': ['.zip', '.rar', '.7z', '.tar', '.gz'] if self.download_other else [],
            'Audio': ['.mp3', '.wav', '.ogg'] if self.download_other else [],
            'Video': ['.mp4', '.avi', '.mov', '.mkv'] if self.download_other else []
        }
        self.downloadable_extensions = {k: v for k, v in self.downloadable_extensions.items() if v}
        self.all_downloadable_exts = {ext for exts in self.downloadable_extensions.values() for ext in exts}

        self.content_type_mapping = {
            'PDF': {'application/pdf': '.pdf'},
            'Image': {
                'image/jpeg': '.jpg',
                'image/png': '.png',
                'image/gif': '.gif',
                'image/svg+xml': '.svg',
            },
            'Doc': {
                'application/msword': '.doc',
                'application/vnd.openxmlformats-officedocument.wordprocessingml.document': '.docx',
                'application/vnd.ms-excel': '.xls',
                'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': '.xlsx',
                'application/vnd.ms-powerpoint': '.ppt',
                'application/vnd.openxmlformats-officedocument.presentationml.presentation': '.pptx'
            },
            'Archive': {
                'application/zip': '.zip',
                'application/x-rar-compressed': '.rar',
                'application/x-7z-compressed': '.7z',
                'application/gzip': '.gz',
                'application/x-tar': '.tar'
            },
            'Audio': {
                'audio/mpeg': '.mp3',
                'audio/wav': '.wav',
                'audio/ogg': '.ogg'
            },
            'Video': {
                'video/mp4': '.mp4',
                'video/x-msvideo': '.avi',
                'video/quicktime': '.mov',
                'video/x-matroska': '.mkv'
            }
        }

        self.session = self.setup_session()
        self.html_converter = html2text.HTML2Text()
        self.html_converter.ignore_links = False
        self.html_converter.body_width = 0
        self.html_converter.ignore_images = True
        self.html_converter.single_line_break = False

        self.language_path = re.search(r'/(fr|en)-(ca|us)/', self.start_url)
        self.language_pattern = self.language_path.group(0) if self.language_path else None

        self.create_directories()

        self.playwright = None
        self.browser = None
        self.page = None
        if self.use_playwright:
            try:
                from playwright.sync_api import sync_playwright
                self.playwright = sync_playwright().start()
                self.browser = self.playwright.chromium.launch(headless=True)
                self.page = self.browser.new_page()
                self.logger.info("Playwright initialisé avec succès.")
            except Exception as e:
                self.logger.error(f"Échec de l'initialisation de Playwright : {str(e)}")
                self._update_metrics('errors', f"Échec de l'initialisation de Playwright : {str(e)}")
                self.use_playwright = False

        self.stop_event = threading.Event()

        system_prompt_file = llm_conf.system_prompt_file
        if system_prompt_file:
            try:
                with open(system_prompt_file, 'r', encoding='utf-8') as f:
                    system_prompt = f.read().strip()
                self.logger.info(f"Prompt système chargé depuis {system_prompt_file}")
            except Exception as e:
                self.logger.error(f"Erreur lors du chargement du fichier prompt {system_prompt_file} : {str(e)}")
                self._update_metrics('errors', f"Erreur lors du chargement du fichier prompt {system_prompt_file} : {str(e)}")
                system_prompt = (
                    "You are an expert content rewriter. Reformulate the text below to improve its clarity and readability while preserving the original meaning."
                )
                self.logger.info("Utilisation du prompt système par défaut.")
        else:
            system_prompt = (
                "You are an expert content rewriter. Reformulate the text below to improve its clarity and readability while preserving the original meaning."
            )
            self.logger.info("Aucun fichier prompt spécifié, utilisation du prompt système par défaut.")

        rewriter_output_dir = crawler_conf.output.content_rewritten_dir
        self.content_rewriter = ContentRewriter(
            input_dir=str(self.base_dir / crawler_conf.output.content_dir),
            output_dir=str(self.base_dir / rewriter_output_dir),
            api_keys=self.api_keys,
            system_prompt=system_prompt,
            model=self.llm_provider if self.llm_provider else "openai",
            verbose=crawler_conf.verbose,
            max_tokens=crawler_conf.limits.max_tokens,
            concurrency=llm_conf.concurrency
        )

    def setup_session(self) -> requests.Session:
        session = requests.Session()
        retry_strategy = requests.adapters.Retry(
            total=5,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"]
        )
        adapter = requests.adapters.HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        session.verify = True
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) '
                          'Chrome/85.0.4183.102 Safari/537.36'
        })
        return session

    def create_directories(self):
        directories = [
            self.config.output.content_dir,
            self.config.output.PDF,
            self.config.output.Image,
            self.config.output.Doc,
            self.config.output.Archive,
            self.config.output.Audio,
            self.config.output.Video,
            self.config.output.logs_dir,
            self.config.output.content_rewritten_dir
        ]
        for dir_path in directories:
            full_path = self.base_dir / dir_path
            full_path.mkdir(parents=True, exist_ok=True)
            self.logger.debug(f"Répertoire créé ou déjà existant : {full_path}")

    def should_exclude(self, url: str) -> bool:
        return any(excluded in url for excluded in self.excluded_paths)

    def is_same_language(self, url: str) -> bool:
        if not self.language_pattern:
            return True
        return self.language_pattern in url

    def is_downloadable_file(self, url: str) -> bool:
        path = urlparse(url).path.lower()
        if not self.all_downloadable_exts:
            return False
        pattern = re.compile(r'\.(' + '|'.join(ext.strip('.') for ext in self.all_downloadable_exts) + r')(\.[a-z0-9]+)?$', re.IGNORECASE)
        return bool(pattern.search(path))

    def head_or_get(self, url: str) -> Optional[requests.Response]:
        try:
            r = self.session.head(url, allow_redirects=True, timeout=10)
            if r.status_code == 405:
                r = self.session.get(url, allow_redirects=True, timeout=10, stream=True)
            return r
        except:
            try:
                return self.session.get(url, allow_redirects=True, timeout=10, stream=True)
            except:
                return None

    def get_file_type_and_extension(self, url: str, response: requests.Response) -> (Optional[str], Optional[str]):
        if response is None:
            return None, None
        path = urlparse(url).path.lower()
        content_type = response.headers.get('Content-Type', '').lower()

        for file_type, extensions in self.downloadable_extensions.items():
            for ext in extensions:
                pattern = re.compile(re.escape(ext) + r'(\.[a-z0-9]+)?$', re.IGNORECASE)
                if pattern.search(path):
                    return file_type, self.content_type_mapping.get(file_type, {}).get(content_type, ext)

        for file_type, mapping in self.content_type_mapping.items():
            if content_type in mapping:
                return file_type, mapping[content_type]

        return None, None

    def sanitize_filename(self, url: str, extension: str) -> str:
        url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
        filename = url.split('/')[-1] or 'index'
        filename = re.sub(r'[^\w\-_.]', '_', filename)
        name = Path(filename).stem
        if not extension:
            extension = '.txt'
        sanitized = f"{name}_{url_hash}{extension}"
        return sanitized

    def download_file(self, url: str) -> bool:
        if self.stop_event.is_set():
            return False

        response = self.head_or_get(url)
        if not response or response.status_code != 200:
            self.logger.warning(f"Échec de la récupération du fichier à {url}")
            self._update_metrics('errors', f"Échec de la récupération du fichier à {url}")
            return False

        file_type_detected, extension = self.get_file_type_and_extension(url, response)
        if not file_type_detected:
            self.logger.warning(f"Impossible de déterminer le type de fichier pour : {url}")
            self._update_metrics('errors', f"Impossible de déterminer le type de fichier pour : {url}")
            return False

        if file_type_detected not in self.downloadable_extensions:
            self.logger.info(f"Type de fichier {file_type_detected} non activé pour le téléchargement.")
            return False

        self.logger.info(f"Tentative de téléchargement du fichier {file_type_detected} depuis : {url}")
        filename = self.sanitize_filename(url, extension)
        save_path = self.base_dir / file_type_detected / filename

        if save_path.exists():
            self.logger.info(f"Fichier déjà téléchargé, passage : {filename}")
            return False

        try:
            if response.request.method == 'HEAD':
                response = self.session.get(url, stream=True, timeout=20)

            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if self.stop_event.is_set():
                        self.logger.info(f"Téléchargement arrêté pour : {url}")
                        return False
                    if chunk:
                        f.write(chunk)

            self.stats[f'{file_type_detected}_downloaded'] += 1
            self.downloaded_files.add(url)
            self.logger.info(f"Téléchargement réussi de {file_type_detected} : {filename}")
            self._update_metrics('files_downloaded', file_type_detected)
            return True
        except Exception as e:
            self.logger.error(f"Erreur lors du téléchargement de {url} : {str(e)}")
            self._update_metrics('errors', f"Erreur lors du téléchargement de {url} : {str(e)}")
            return False

    def fetch_page_content(self, url: str) -> Optional[str]:
        if self.use_playwright and self.page:
            try:
                self.page.goto(url, timeout=20000)
                time.sleep(2)
                return self.page.content()
            except Exception as e:
                self.logger.error(f"Playwright a échoué à récupérer {url} : {str(e)}")
                self._update_metrics('errors', f"Playwright a échoué à récupérer {url} : {str(e)}")
                return None
        else:
            try:
                response = self.session.get(url, timeout=20)
                if response.status_code == 200:
                    return response.text
                else:
                    self.logger.warning(f"Échec de la récupération de {url}, code de statut : {response.status_code}")
                    self._update_metrics('errors', f"Échec de la récupération de {url}, code de statut : {response.status_code}")
                    return None
            except Exception as e:
                self.logger.error(f"Requests a échoué à récupérer {url} : {str(e)}")
                self._update_metrics('errors', f"Requests a échoué à récupérer {url} : {str(e)}")
                return None

    def convert_links_to_absolute(self, soup: BeautifulSoup, base_url: str) -> BeautifulSoup:
        for tag in soup.find_all(['a', 'embed', 'iframe', 'object'], href=True):
            attr = 'href' if tag.name == 'a' else 'src'
            href = tag.get(attr)
            if href:
                absolute_url = urljoin(base_url, href)
                tag[attr] = absolute_url
        return soup

    def clean_text(self, text: str) -> str:
        if not text:
            return ""
        text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', text)
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n\n', text)
        return text.strip()

    def extract_urls(self, start_url: str):
        queue = deque([(start_url, 0)])
        self.visited_pages.add(start_url)
        crawled_count = 0
        total_estimated = self.max_urls if self.max_urls else 1000

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeRemainingColumn(),
            console=self.console
        ) as progress:
            task = progress.add_task("Extraction des URLs", total=total_estimated)

            while queue:
                if self.stop_event.is_set():
                    self.logger.info("Exploration arrêtée par l'utilisateur.")
                    break

                current_url, depth = queue.popleft()

                if self.max_urls is not None and crawled_count >= self.max_urls:
                    self.logger.info(f"Limite max_urls atteinte ({self.max_urls}), arrêt de l'extraction des URLs.")
                    break

                if self.max_urls is None and depth > self.max_depth:
                    continue

                if self.should_exclude(current_url):
                    self.logger.info(f"URL exclue : {current_url}")
                    progress.advance(task)
                    continue

                self.logger.info(f"Extraction des URLs de : {current_url} (profondeur : {depth})")
                crawled_count += 1
                progress.advance(task)

                if self.is_downloadable_file(current_url):
                    success = self.download_file(current_url)
                    if success:
                        self.stats['files_downloaded'] += 1
                    continue

                page_content = self.fetch_page_content(current_url)
                if page_content is None:
                    self.logger.warning(f"Impossible de récupérer le contenu de : {current_url}")
                    continue

                soup = BeautifulSoup(page_content, 'html.parser')
                child_links = set()
                for tag in soup.find_all(['a', 'link', 'embed', 'iframe', 'object'], href=True):
                    href = tag.get('href') or tag.get('src')
                    if not href:
                        continue
                    absolute_url = urljoin(current_url, href)
                    parsed_url = urlparse(absolute_url)

                    if self.is_downloadable_file(absolute_url):
                        self.download_file(absolute_url)
                        continue

                    if (self.domain in parsed_url.netloc
                            and self.is_same_language(absolute_url)
                            and not absolute_url.endswith(('#', 'javascript:void(0)', 'javascript:;'))
                            and not self.should_exclude(absolute_url)):
                        child_links.add(absolute_url)
                        if absolute_url not in self.visited_pages:
                            if self.max_urls is None or crawled_count < self.max_urls:
                                if self.max_urls is None and depth + 1 > self.max_depth:
                                    continue
                                queue.append((absolute_url, depth + 1))
                                self.visited_pages.add(absolute_url)

                self.site_map[current_url].update(child_links)
                self.stats['urls_found'] += len(child_links)
                self._update_metrics('urls_extracted', len(child_links))

        self.logger.info("Extraction des URLs terminée.")

    def extract_content(self, url: str):
        if self.is_downloadable_file(url):
            self.logger.debug(f"Passage de l'extraction du contenu pour le fichier téléchargeable : {url}")
            return

        page_content = self.fetch_page_content(url)
        if page_content is None:
            self.logger.warning(f"Impossible de récupérer le contenu pour : {url}")
            return

        soup = BeautifulSoup(page_content, 'html.parser')
        for element in soup.find_all(['nav', 'header', 'footer', 'script', 'style', 'aside', 'iframe']):
            element.decompose()

        main_content = (soup.find('main') or soup.find('article') or 
                        soup.find('div', class_='content') or soup.find('div', id='content') or
                        soup.find('section', class_='main-section'))

        if not main_content:
            self.logger.warning(f"Aucun contenu principal trouvé pour : {url}")
            return

        self.convert_links_to_absolute(main_content, url)
        markdown_content = self.html_converter.handle(str(main_content))

        title = soup.find('h1')
        content_parts = []
        if title:
            content_parts.append(f"# {title.get_text().strip()}")
        content_parts.append(f"**Source :** {url}")
        content_parts.append(markdown_content)

        content = self.clean_text('\n\n'.join(content_parts))

        if content:
            filename = self.sanitize_filename(url, '.txt')
            save_path = self.base_dir / self.config.output.content_dir / filename
            try:
                with open(save_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                self.stats['pages_processed'] += 1
                self._update_metrics('pages_extracted', 1)

                rewrite_success = self.content_rewriter.rewrite_file(save_path)
                if rewrite_success:
                    self.logger.info(f"Réécriture du contenu réussie pour {filename}")
                else:
                    self.logger.warning(f"Réécriture du contenu échouée pour {filename}")

            except Exception as e:
                self.logger.error(f"Erreur lors de la sauvegarde du contenu pour {url} : {str(e)}")
                self._update_metrics('errors', f"Erreur lors de la sauvegarde du contenu pour {url} : {str(e)}")
        else:
            self.logger.warning(f"Aucun contenu significatif trouvé pour : {url}")

        for tag in main_content.find_all(['a', 'embed', 'iframe', 'object'], href=True):
            href = tag.get('href') or tag.get('src')
            if href:
                file_url = urljoin(url, href)
                if self.is_downloadable_file(file_url) and file_url not in self.downloaded_files:
                    self.download_file(file_url)

    def load_downloaded_files(self):
        downloaded_files_path = self.base_dir / self.config.output.logs_dir / 'downloaded_files.txt'
        if downloaded_files_path.exists():
            try:
                with open(downloaded_files_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        self.downloaded_files.add(line.strip())
                self.logger.info(f"{len(self.downloaded_files)} fichiers téléchargés chargés.")
            except Exception as e:
                self.logger.error(f"Erreur lors du chargement du suivi des fichiers téléchargés : {str(e)}")
                self._update_metrics('errors', f"Erreur lors du chargement du suivi des fichiers téléchargés : {str(e)}")
        else:
            self.logger.info("Aucun fichier de suivi des téléchargements trouvé, démarrage sans historique.")

    def save_downloaded_files(self):
        downloaded_files_path = self.base_dir / self.config.output.logs_dir / 'downloaded_files.txt'
        try:
            with open(downloaded_files_path, 'w', encoding='utf-8') as f:
                for url in sorted(self.downloaded_files):
                    f.write(url + '\n')
            self.logger.info(f"{len(self.downloaded_files)} fichiers téléchargés sauvegardés.")
        except Exception as e:
            self.logger.error(f"Erreur lors de la sauvegarde du suivi des fichiers téléchargés : {str(e)}")
            self._update_metrics('errors', f"Erreur lors de la sauvegarde du suivi des fichiers téléchargés : {str(e)}")

    def generate_report(self, duration: float, error: Optional[str] = None):
        report_data = {
            "configuration": {
                "start_url": self.start_url,
                "language_pattern": self.language_pattern,
                "max_depth": self.max_depth,
                "max_urls": self.max_urls,
                "duration_seconds": duration
            },
            "statistics": {
                "total_urls_found": self.stats.get('urls_found', 0),
                "pages_processed": self.stats.get('pages_processed', 0),
                "files_downloaded": {k: self.stats.get(f"{k}_downloaded", 0) for k in self.downloadable_extensions.keys()},
                "total_files_downloaded": sum(self.stats.get(f"{k}_downloaded", 0) for k in self.downloadable_extensions.keys())
            },
            "status": "Terminé avec des erreurs" if error else "Terminé avec succès",
            "visited_pages": sorted(self.visited_pages),
            "downloaded_files": sorted(self.downloaded_files),
            "errors": metrics['errors'] if metrics['errors'] else None
        }

        json_report_path = self.base_dir / 'report.json'
        try:
            with open(json_report_path, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=4, ensure_ascii=False)
            self.logger.info(f"Rapport JSON généré avec succès : {json_report_path}")
        except Exception as e:
            self.logger.error(f"Erreur lors de la génération du rapport JSON : {str(e)}")
            self._update_metrics('errors', f"Erreur lors de la génération du rapport JSON : {str(e)}")

    def generate_xml_sitemap(self):
        visited = set()

        def add_page_element(parent_elem, url):
            if url in visited:
                return
            visited.add(url)
            page_elem = SubElement(parent_elem, "page", url=url)
            for child_url in sorted(self.site_map[url]):
                add_page_element(page_elem, child_url)

        root = Element("site", start_url=self.start_url)
        add_page_element(root, self.start_url)

        tree = ElementTree(root)
        xml_path = self.base_dir / 'sitemap.xml'
        try:
            tree.write(xml_path, encoding='utf-8', xml_declaration=True)
            self.logger.info(f"Sitemap XML générée avec succès : {xml_path}")
        except Exception as e:
            self.logger.error(f"Erreur lors de la génération de la sitemap XML : {str(e)}")
            self._update_metrics('errors', f"Erreur lors de la génération de la sitemap XML : {str(e)}")

    def save_checkpoint(self):
        checkpoint_data = {
            "visited_pages": list(self.visited_pages),
            "downloaded_files": list(self.downloaded_files),
            "site_map": {k: list(v) for k, v in self.site_map.items()},
            "stats": dict(self.stats)
        }
        try:
            with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, ensure_ascii=False, indent=4)
            self.logger.info(f"Checkpoint sauvegardé : {self.checkpoint_file}")
        except Exception as e:
            self.logger.error(f"Erreur lors de la sauvegarde du checkpoint : {str(e)}")
            self._update_metrics('errors', f"Erreur lors de la sauvegarde du checkpoint : {str(e)}")

    def load_checkpoint(self):
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                    checkpoint_data = json.load(f)
                self.visited_pages = set(checkpoint_data.get("visited_pages", []))
                self.downloaded_files = set(checkpoint_data.get("downloaded_files", []))
                self.site_map = {k: set(v) for k, v in checkpoint_data.get("site_map", {}).items()}
                self.stats = defaultdict(int, checkpoint_data.get("stats", {}))
                self.logger.info(f"Checkpoint chargé : {self.checkpoint_file}")
            except Exception as e:
                self.logger.error(f"Erreur lors du chargement du checkpoint : {str(e)}")
                self._update_metrics('errors', f"Erreur lors du chargement du checkpoint : {str(e)}")

    def crawl(self):
        start_time = time.time()
        self.logger.info(f"Démarrage de l'exploration de {self.start_url}")

        self.load_downloaded_files()
        self.load_checkpoint()
        error = None
        try:
            self.logger.info("Phase 1 : Démarrage de l'extraction des URLs")
            if not self.visited_pages:
                self.extract_urls(self.start_url)
                self.save_checkpoint()
            else:
                self.logger.info("Checkpoint trouvé, passage de la phase d'extraction des URLs.")

            self.logger.info("Phase 2 : Démarrage de l'extraction du contenu")
            total_pages = len(self.visited_pages)
            if total_pages == 0:
                self.logger.info("Aucune page à traiter.")
            else:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    "[progress.percentage]{task.percentage:>3.0f}%",
                    TimeRemainingColumn(),
                    console=self.console
                ) as progress:
                    task = progress.add_task("Extraction du contenu", total=total_pages)

                    with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
                        futures = {executor.submit(self.extract_content, url): url for url in self.visited_pages}
                        for future in as_completed(futures):
                            url = futures[future]
                            try:
                                future.result()
                                progress.advance(task)
                            except Exception as e:
                                self.logger.error(f"Erreur lors de l'extraction de {url} : {str(e)}")
                                self._update_metrics('errors', f"Erreur lors de l'extraction de {url} : {str(e)}")
                                progress.advance(task)
                self.logger.info("Phase 2 : Extraction du contenu terminée.")

            self.save_checkpoint()

            if self.llm_enabled:
                self.logger.info("Phase 3 : Démarrage de la réécriture du contenu")
                self.content_rewriter.rewrite_all_contents()

            end_time = time.time()
            duration = end_time - start_time
            self.generate_report(duration, error=error)
            self.generate_xml_sitemap()

        except Exception as e:
            error = str(e)
            self.logger.error(f"Erreur critique durant l'exploration : {str(e)}")
            self._update_metrics('errors', f"Erreur critique durant l'exploration : {str(e)}")

        self.save_downloaded_files()
        if self.use_playwright and self.page:
            self.page.close()
            self.browser.close()
            self.playwright.stop()

    def stop_crawl(self):
        self.logger.info("Signal d'arrêt reçu. Arrêt du crawler...")
        self.stop_event.set()

    def _update_metrics(self, key: str, value):
        with metrics_lock:
            if key == 'errors':
                metrics['errors'].append(value)
            elif key == 'files_downloaded' and isinstance(value, str):
                metrics['files_downloaded'][value] += 1
            elif key in metrics:
                metrics[key] += value
            else:
                metrics[key] = value


def load_config(config_path: str = 'config.json') -> dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Le fichier de configuration {config_path} est introuvable.")
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        return config_data
    except json.JSONDecodeError as e:
        raise ValueError(f"Erreur de décodage JSON dans {config_path} : {e}")


def main():
    try:
        config_data = load_config()
        config = Config.parse_obj(config_data)
    except (FileNotFoundError, ValueError, ValidationError) as e:
        print(f"Erreur de configuration : {e}")
        return

    crawler = WebCrawler(config.dict())

    try:
        crawler.crawl()
    except KeyboardInterrupt:
        crawler.stop_crawl()
        crawler.save_checkpoint()
        crawler.save_downloaded_files()
        if crawler.use_playwright and crawler.page:
            crawler.page.close()
            crawler.browser.close()
            crawler.playwright.stop()
        crawler.logger.info("Crawler arrêté par l'utilisateur.")


if __name__ == "__main__":
    main()
