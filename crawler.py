#!/usr/bin/env python3
# crawler.py

"""
Author: Simon-Pierre Boucher
Université Laval, Québec
Date: 2024-12-10

Description:
    This script implements a web crawler capable of extracting URLs, downloading files,
    extracting content from web pages, and rewriting this content using a specified
    large language model (LLM) provider. It supports error handling, metric tracking,
    and generating detailed reports at the end of execution.
"""

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

# Suppress insecure SSL warnings
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Configuration of global variables
CONTENT_REWRITER_OUTPUT_DIR = "./output/content_rewritten"

# Mutex to synchronize access to metrics
metrics_lock = threading.Lock()

# Initialize metrics
metrics = {
    "urls_extracted": 0,
    "pages_extracted": 0,
    "pages_rewritten": 0,
    "files_downloaded": defaultdict(int),
    "errors": []
}


class ContentRewriter:
    """
    Class responsible for rewriting content using a specified LLM provider.

    Attributes:
        input_dir (Path): Directory containing input text files.
        output_dir (Path): Directory where rewritten content will be saved.
        api_keys (List[str]): List of API keys for the LLM provider.
        system_prompt (str): System prompt loaded from an external file.
        model (str): LLM model to use.
        verbose (bool): Indicator to enable detailed logging.
        max_tokens (int): Maximum number of tokens per API request.
        concurrency (int): Number of concurrent threads for processing.
    """

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
        """
        Initializes the ContentRewriter with the specified parameters.

        Args:
            input_dir (str): Directory containing input text files.
            output_dir (str): Directory where rewritten content will be saved.
            api_keys (List[str]): List of API keys for the LLM provider.
            system_prompt (str): System prompt loaded from an external file.
            model (str, optional): LLM model to use. Defaults to "openai".
            verbose (bool, optional): Enable detailed logging. Defaults to False.
            max_tokens (int, optional): Maximum number of tokens per API request. Defaults to 2048.
            concurrency (int, optional): Number of concurrent threads for processing. Defaults to 5.
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.api_keys = api_keys if api_keys else []
        if not self.api_keys:
            raise ValueError("At least one API key is required for content rewriting.")
        self.api_keys_cycle = cycle(self.api_keys)
        self.lock = threading.Lock()
        
        self.model = model.lower()
        self.verbose = verbose
        self.max_tokens = max_tokens  # Token limit per request
        self.concurrency = concurrency  # Number of concurrent threads
        
        # Store the system prompt
        self.system_prompt = system_prompt
        
        # Configure logging with Rich
        logging.basicConfig(
            level=logging.INFO if not verbose else logging.DEBUG,
            format='%(message)s',
            datefmt="[%X]",
            handlers=[RichHandler()]
        )
        self.logger = logging.getLogger(__name__)

        self.console = Console()

    def get_next_api_key(self) -> Optional[str]:
        """
        Retrieves the next available API key from the cyclic list.

        Returns:
            Optional[str]: The next API key or None if unavailable.
        """
        with self.lock:
            try:
                return next(self.api_keys_cycle)
            except StopIteration:
                return None

    def split_text_into_chunks(self, text: str) -> List[str]:
        """
        Splits the text into manageable chunks based on the token limit.

        Args:
            text (str): The text to split.

        Returns:
            List[str]: List of text chunks.
        """
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
        
        self.logger.debug(f"Text split into {len(chunks)} chunks.")
        return chunks

    def rewrite_chunk(self, chunk: str, document_name: str, file_name: str) -> Optional[str]:
        """
        Rewrites a chunk of text using the specified LLM provider.

        Args:
            chunk (str): The text chunk to rewrite.
            document_name (str): Name of the source document.
            file_name (str): Name of the source file.

        Returns:
            Optional[str]: The rewritten text or None in case of an error.
        """
        system_prompt = self.system_prompt
        user_content = chunk

        api_key = self.get_next_api_key()
        if not api_key:
            self.logger.error("No API key available for rewriting.")
            self._update_metrics('errors', "No API key available for rewriting.")
            return None

        if self.model == "openai":
            endpoint = "https://api.openai.com/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": "gpt-4",
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
            self.logger.error(f"Unknown LLM provider: {self.model}")
            self._update_metrics('errors', f"Unknown LLM provider: {self.model}")
            return None

        if self.verbose:
            self.logger.debug(f"Calling LLM {self.model} for {document_name}: Payload size {len(user_content)}")

        try:
            response = requests.post(endpoint, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
        except requests.HTTPError as e:
            self.logger.error(f"LLM API error {self.model}: {str(e)}, response: {response.text}")
            self._update_metrics('errors', f"LLM API error {self.model}: {str(e)}, response: {response.text}")
            return None
        except Exception as e:
            self.logger.error(f"Error calling LLM {self.model}: {str(e)}")
            self._update_metrics('errors', f"Error calling LLM {self.model}: {str(e)}")
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
        """
        Rewrites the content of a file using the ContentRewriter.

        Args:
            file_path (Path): Path to the file to rewrite.

        Returns:
            bool: True if rewriting was successful, False otherwise.
        """
        document_name = file_path.stem
        self.logger.info(f"Rewriting content of {file_path.name}")

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        except Exception as e:
            self.logger.error(f"Error reading {file_path.name}: {str(e)}")
            self._update_metrics('errors', f"Error reading {file_path.name}: {str(e)}")
            return False

        if not text.strip():
            self.logger.warning(f"No text to rewrite in {file_path.name}")
            self._update_metrics('errors', f"No text to rewrite in {file_path.name}")
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
            task = progress.add_task(f"Rewriting {file_path.name}", total=total_chunks)

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
                        self.logger.error(f"Error rewriting segment {idx + 1}: {str(e)}")
                        self._update_metrics('errors', f"Error rewriting segment {idx + 1}: {str(e)}")

        if any(chunk is None for chunk in rewritten_chunks):
            self.logger.warning(f"Partial rewrite failed for {file_path.name}")
            return False

        rewritten_text = "\n\n".join(rewritten_chunks)
        output_file_name = self.output_dir / f"{document_name}_rewritten.txt"
        try:
            with open(output_file_name, 'w', encoding='utf-8') as f:
                f.write(rewritten_text)
            self.logger.info(f"Rewritten file created: {output_file_name}")
            self._update_metrics('pages_rewritten', 1)
            return True
        except Exception as e:
            self.logger.error(f"Error saving {file_path}: {str(e)}")
            self._update_metrics('errors', f"Error saving {file_path}: {str(e)}")
            return False

    def rewrite_all_contents(self):
        """
        Rewrites the content of all files in the input directory.
        """
        txt_files = list(self.input_dir.glob('*.txt'))
        total_files = len(txt_files)
        self.logger.info(f"Starting to rewrite {total_files} file(s)")

        if total_files == 0:
            self.logger.info("No files to rewrite.")
            return

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeRemainingColumn(),
            transient=True,
            console=self.console
        ) as progress:
            task = progress.add_task("Rewriting files", total=total_files)

            with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
                futures = {executor.submit(self.rewrite_file, txt_file_path): txt_file_path for txt_file_path in txt_files}

                for future in as_completed(futures):
                    txt_file_path = futures[future]
                    try:
                        result = future.result()
                        if result:
                            self.logger.info(f"Successfully rewritten {txt_file_path.name}")
                        progress.advance(task)
                    except Exception as e:
                        self.logger.error(f"Error rewriting {txt_file_path.name}: {str(e)}")
                        self._update_metrics('errors', f"Error rewriting {txt_file_path.name}: {str(e)}")
                        progress.advance(task)
        self.logger.info("Content rewriting completed.")

    def _update_metrics(self, key: str, value):
        """
        Updates global metrics in a thread-safe manner.

        Args:
            key (str): The metric key to update.
            value: The value to add or increment.
        """
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
    """
    Configuration for the types of files to download.

    Attributes:
        pdf (bool): Download PDF files.
        doc (bool): Download Word, Excel, etc., documents.
        image (bool): Download images.
        other (bool): Download other types of files.
    """
    pdf: bool = False
    doc: bool = False
    image: bool = False
    other: bool = False


class OutputConfig(BaseModel):
    """
    Configuration for output directories.

    Attributes:
        crawler_output_dir (str): Main output directory for the crawler.
        checkpoint_file (str): Checkpoint file for resuming.
        logs_dir (str): Directory for logs.
        content_dir (str): Directory for extracted content.
        content_rewritten_dir (str): Directory for rewritten content.
        PDF (str): Directory for downloaded PDF files.
        Image (str): Directory for downloaded images.
        Doc (str): Directory for downloaded documents.
        Archive (str): Directory for downloaded archives.
        Audio (str): Directory for downloaded audio files.
        Video (str): Directory for downloaded video files.
    """
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
    """
    Configuration for the crawler's operational limits.

    Attributes:
        max_tokens (int): Maximum number of tokens per API request.
        concurrency (int): Number of concurrent threads for processing.
        max_urls (Optional[int]): Maximum number of URLs to crawl.
    """
    max_tokens: int = 2048
    concurrency: int = 5
    max_urls: Optional[int] = None


class CrawlerConfig(BaseModel):
    """
    General configuration for the crawler.

    Attributes:
        start_url (str): Starting URL for crawling.
        max_depth (int): Maximum crawling depth.
        use_playwright (bool): Use Playwright for page rendering.
        download (DownloadConfig): Configuration for types of files to download.
        output (OutputConfig): Configuration for output directories.
        limits (LimitsConfig): Configuration for operational limits.
        verbose (bool): Enable detailed logging.
    """
    start_url: str
    max_depth: int = 2
    use_playwright: bool = False
    download: DownloadConfig
    output: OutputConfig
    limits: LimitsConfig
    verbose: bool = False


class LLMConfig(BaseModel):
    """
    Configuration for the LLM provider.

    Attributes:
        provider (str): LLM provider to use (e.g., "openai", "anthropic", "mistral").
        api_keys (List[str]): List of API keys for the LLM provider.
        max_tokens (int): Maximum number of tokens per API request.
        concurrency (int): Number of concurrent threads for processing.
        system_prompt_file (str): Path to the file containing the system prompt.
    """
    provider: str = "openai"
    api_keys: List[str]
    max_tokens: int = 2048
    concurrency: int = 5
    system_prompt_file: str


class Config(BaseModel):
    """
    Complete configuration for the crawler and content rewriter.

    Attributes:
        crawler (CrawlerConfig): Configuration for the crawler.
        llm (LLMConfig): Configuration for the LLM provider.
    """
    crawler: CrawlerConfig
    llm: LLMConfig


class WebCrawler:
    """
    Class representing the web crawler responsible for extracting URLs, downloading files,
    extracting content, and rewriting content using an LLM.

    Methods:
        __init__: Initializes the crawler with the given configuration.
        setup_session: Configures the HTTP session with retry strategies.
        create_directories: Creates necessary directories for the crawler to operate.
        should_exclude: Determines if a URL should be excluded from crawling.
        is_same_language: Checks if a URL matches the specified language pattern.
        is_downloadable_file: Determines if a URL points to a downloadable file.
        head_or_get: Performs a HEAD or GET request to obtain headers or content.
        get_file_type_and_extension: Determines the file type and its extension based on URL and headers.
        sanitize_filename: Generates a secure filename from a URL.
        download_file: Downloads a file from a URL.
        fetch_page_content: Retrieves the HTML content of a web page.
        convert_links_to_absolute: Converts relative links to absolute links in HTML content.
        clean_text: Cleans text by removing unwanted characters.
        extract_urls: Extracts URLs starting from the initial URL up to a maximum depth.
        extract_content: Extracts the main content from a page and saves it.
        load_downloaded_files: Loads the history of downloaded files.
        save_downloaded_files: Saves the history of downloaded files.
        generate_report: Generates a JSON report of the operations performed.
        generate_xml_sitemap: Generates an XML sitemap of the crawled URLs.
        save_checkpoint: Saves the current state of the crawler for later resumption.
        load_checkpoint: Loads a previous state of the crawler to resume crawling.
        crawl: Starts the crawling process.
        stop_crawl: Stops the crawling process.
        _update_metrics: Updates global metrics in a thread-safe manner.
    """

    def __init__(self, config: dict):
        """
        Initializes the WebCrawler with the given configuration.

        Args:
            config (dict): Configuration dictionary.
        """
        # Load configuration with Pydantic
        try:
            self.config = CrawlerConfig.parse_obj(config['crawler'])
            self.llm_config = LLMConfig.parse_obj(config['llm'])
        except ValidationError as e:
            print("Configuration validation error:", e)
            raise

        # Extract configurations
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

        # Configure logging with Rich
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

        # Extract language pattern from the starting URL (optional)
        self.language_path = re.search(r'/(fr|en)-(ca|us)/', self.start_url)
        self.language_pattern = self.language_path.group(0) if self.language_path else None

        # Create necessary directories
        self.create_directories()

        # Initialize Playwright if enabled
        self.playwright = None
        self.browser = None
        self.page = None
        if self.use_playwright:
            try:
                from playwright.sync_api import sync_playwright
                self.playwright = sync_playwright().start()
                self.browser = self.playwright.chromium.launch(headless=True)
                self.page = self.browser.new_page()
                self.logger.info("Playwright initialized successfully.")
            except Exception as e:
                self.logger.error(f"Failed to initialize Playwright: {str(e)}")
                self._update_metrics('errors', f"Failed to initialize Playwright: {str(e)}")
                self.use_playwright = False  # Disable Playwright if initialization fails

        self.stop_event = threading.Event()

        # Load the system prompt from the specified file
        system_prompt_file = llm_conf.system_prompt_file
        if system_prompt_file:
            try:
                with open(system_prompt_file, 'r', encoding='utf-8') as f:
                    system_prompt = f.read().strip()
                self.logger.info(f"System prompt loaded from {system_prompt_file}")
            except Exception as e:
                self.logger.error(f"Error loading prompt file {system_prompt_file}: {str(e)}")
                self._update_metrics('errors', f"Error loading prompt file {system_prompt_file}: {str(e)}")
                system_prompt = (
                    "You are an expert content rewriter. Reformulate the text below to improve its clarity and readability while preserving the original meaning."
                )
                self.logger.info("Using default system prompt.")
        else:
            system_prompt = (
                "You are an expert content rewriter. Reformulate the text below to improve its clarity and readability while preserving the original meaning."
            )
            self.logger.info("No prompt file specified, using default system prompt.")

        # Instantiate the ContentRewriter
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
        """
        Configures the HTTP session with retry strategies.

        Returns:
            requests.Session: The configured HTTP session.
        """
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
        session.verify = True  # Enable SSL verification
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) '
                          'Chrome/85.0.4183.102 Safari/537.36'
        })
        return session

    def create_directories(self):
        """
        Creates the necessary directories for the crawler to operate.
        """
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
            self.logger.debug(f"Directory created or already exists: {full_path}")

    def should_exclude(self, url: str) -> bool:
        """
        Determines if a URL should be excluded from crawling.

        Args:
            url (str): The URL to evaluate.

        Returns:
            bool: True if the URL should be excluded, False otherwise.
        """
        return any(excluded in url for excluded in self.excluded_paths)

    def is_same_language(self, url: str) -> bool:
        """
        Checks if a URL matches the specified language pattern.

        Args:
            url (str): The URL to evaluate.

        Returns:
            bool: True if the URL matches the language pattern, False otherwise.
        """
        if not self.language_pattern:
            return True
        return self.language_pattern in url

    def is_downloadable_file(self, url: str) -> bool:
        """
        Determines if a URL points to a downloadable file.

        Args:
            url (str): The URL to evaluate.

        Returns:
            bool: True if it's a downloadable file, False otherwise.
        """
        path = urlparse(url).path.lower()
        if not self.all_downloadable_exts:
            return False
        pattern = re.compile(r'\.(' + '|'.join(ext.strip('.') for ext in self.all_downloadable_exts) + r')(\.[a-z0-9]+)?$', re.IGNORECASE)
        return bool(pattern.search(path))

    def head_or_get(self, url: str) -> Optional[requests.Response]:
        """
        Performs a HEAD or GET request to obtain headers or content of a URL.

        Args:
            url (str): The URL to request.

        Returns:
            Optional[requests.Response]: The HTTP response or None on failure.
        """
        try:
            r = self.session.head(url, allow_redirects=True, timeout=10)
            if r.status_code == 405:  # Method not allowed, try GET
                r = self.session.get(url, allow_redirects=True, timeout=10, stream=True)
            return r
        except:
            try:
                return self.session.get(url, allow_redirects=True, timeout=10, stream=True)
            except:
                return None

    def get_file_type_and_extension(self, url: str, response: requests.Response) -> (Optional[str], Optional[str]):
        """
        Determines the file type and its extension based on the URL and response headers.

        Args:
            url (str): The URL of the file.
            response (requests.Response): The associated HTTP response.

        Returns:
            tuple(Optional[str], Optional[str]): The file type and its extension.
        """
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
        """
        Generates a secure filename from a URL.

        Args:
            url (str): The original URL.
            extension (str): The file extension.

        Returns:
            str: The sanitized filename.
        """
        url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
        filename = url.split('/')[-1] or 'index'
        filename = re.sub(r'[^\w\-_.]', '_', filename)
        name = Path(filename).stem
        if not extension:
            extension = '.txt'
        sanitized = f"{name}_{url_hash}{extension}"
        return sanitized

    def download_file(self, url: str) -> bool:
        """
        Downloads a file from a URL.

        Args:
            url (str): The URL of the file to download.

        Returns:
            bool: True if the download was successful, False otherwise.
        """
        if self.stop_event.is_set():
            return False

        response = self.head_or_get(url)
        if not response or response.status_code != 200:
            self.logger.warning(f"Failed to retrieve file from {url}")
            self._update_metrics('errors', f"Failed to retrieve file from {url}")
            return False

        file_type_detected, extension = self.get_file_type_and_extension(url, response)
        if not file_type_detected:
            self.logger.warning(f"Unable to determine file type for: {url}")
            self._update_metrics('errors', f"Unable to determine file type for: {url}")
            return False

        if file_type_detected not in self.downloadable_extensions:
            self.logger.info(f"File type {file_type_detected} not enabled for download.")
            return False

        self.logger.info(f"Attempting to download {file_type_detected} from: {url}")
        filename = self.sanitize_filename(url, extension)
        save_path = self.base_dir / file_type_detected / filename

        if save_path.exists():
            self.logger.info(f"File already downloaded, skipping: {filename}")
            return False

        try:
            if response.request.method == 'HEAD':
                response = self.session.get(url, stream=True, timeout=20)

            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if self.stop_event.is_set():
                        self.logger.info(f"Download stopped for: {url}")
                        return False
                    if chunk:
                        f.write(chunk)

            self.stats[f'{file_type_detected}_downloaded'] += 1
            self.downloaded_files.add(url)
            self.logger.info(f"Successfully downloaded {file_type_detected}: {filename}")
            self._update_metrics('files_downloaded', file_type_detected)
            return True
        except Exception as e:
            self.logger.error(f"Error downloading {url}: {str(e)}")
            self._update_metrics('errors', f"Error downloading {url}: {str(e)}")
            return False

    def fetch_page_content(self, url: str) -> Optional[str]:
        """
        Retrieves the HTML content of a web page.

        Args:
            url (str): The URL of the page to retrieve.

        Returns:
            Optional[str]: The HTML content of the page or None on failure.
        """
        if self.use_playwright and self.page:
            try:
                self.page.goto(url, timeout=20000)
                time.sleep(2)
                return self.page.content()
            except Exception as e:
                self.logger.error(f"Playwright failed to retrieve {url}: {str(e)}")
                self._update_metrics('errors', f"Playwright failed to retrieve {url}: {str(e)}")
                return None
        else:
            try:
                response = self.session.get(url, timeout=20)
                if response.status_code == 200:
                    return response.text
                else:
                    self.logger.warning(f"Failed to retrieve {url}, status code: {response.status_code}")
                    self._update_metrics('errors', f"Failed to retrieve {url}, status code: {response.status_code}")
                    return None
            except Exception as e:
                self.logger.error(f"Requests failed to retrieve {url}: {str(e)}")
                self._update_metrics('errors', f"Requests failed to retrieve {url}: {str(e)}")
                return None

    def convert_links_to_absolute(self, soup: BeautifulSoup, base_url: str) -> BeautifulSoup:
        """
        Converts relative links to absolute links in the HTML content.

        Args:
            soup (BeautifulSoup): The BeautifulSoup object containing the HTML content.
            base_url (str): The base URL to convert relative links.

        Returns:
            BeautifulSoup: The updated BeautifulSoup object with absolute links.
        """
        for tag in soup.find_all(['a', 'embed', 'iframe', 'object'], href=True):
            attr = 'href' if tag.name == 'a' else 'src'
            href = tag.get(attr)
            if href:
                absolute_url = urljoin(base_url, href)
                tag[attr] = absolute_url
        return soup

    def clean_text(self, text: str) -> str:
        """
        Cleans text by removing unwanted characters.

        Args:
            text (str): The text to clean.

        Returns:
            str: The cleaned text.
        """
        if not text:
            return ""
        text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', text)
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n\n', text)
        return text.strip()

    def extract_urls(self, start_url: str):
        """
        Extracts URLs starting from the initial URL up to a maximum depth.

        Args:
            start_url (str): The starting URL for crawling.
        """
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
            task = progress.add_task("Extracting URLs", total=total_estimated)

            while queue:
                if self.stop_event.is_set():
                    self.logger.info("Crawling stopped by user.")
                    break

                current_url, depth = queue.popleft()

                if self.max_urls is not None and crawled_count >= self.max_urls:
                    self.logger.info(f"Max URLs limit reached ({self.max_urls}), stopping URL extraction.")
                    break

                if self.max_urls is None and depth > self.max_depth:
                    continue

                if self.should_exclude(current_url):
                    self.logger.info(f"Excluded URL: {current_url}")
                    progress.advance(task)
                    continue

                self.logger.info(f"Extracting URLs from: {current_url} (depth: {depth})")
                crawled_count += 1
                progress.advance(task)

                if self.is_downloadable_file(current_url):
                    success = self.download_file(current_url)
                    if success:
                        self.stats['files_downloaded'] += 1
                    continue

                page_content = self.fetch_page_content(current_url)
                if page_content is None:
                    self.logger.warning(f"Unable to retrieve content from: {current_url}")
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

        self.logger.info("URL extraction completed.")

    def extract_content(self, url: str):
        """
        Extracts the main content from a page and saves it.

        Args:
            url (str): The URL of the page to process.
        """
        if self.is_downloadable_file(url):
            self.logger.debug(f"Skipping content extraction for downloadable file: {url}")
            return

        page_content = self.fetch_page_content(url)
        if page_content is None:
            self.logger.warning(f"Unable to retrieve content for: {url}")
            return

        soup = BeautifulSoup(page_content, 'html.parser')
        for element in soup.find_all(['nav', 'header', 'footer', 'script', 'style', 'aside', 'iframe']):
            element.decompose()

        main_content = (soup.find('main') or soup.find('article') or 
                        soup.find('div', class_='content') or soup.find('div', id='content') or
                        soup.find('section', class_='main-section'))

        if not main_content:
            self.logger.warning(f"No main content found for: {url}")
            return

        self.convert_links_to_absolute(main_content, url)
        markdown_content = self.html_converter.handle(str(main_content))

        title = soup.find('h1')
        content_parts = []
        if title:
            content_parts.append(f"# {title.get_text().strip()}")
        content_parts.append(f"**Source:** {url}")
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

                # Call the ContentRewriter immediately
                rewrite_success = self.content_rewriter.rewrite_file(save_path)
                if rewrite_success:
                    self.logger.info(f"Successfully rewritten content for {filename}")
                else:
                    self.logger.warning(f"Failed to rewrite content for {filename}")

            except Exception as e:
                self.logger.error(f"Error saving content for {url}: {str(e)}")
                self._update_metrics('errors', f"Error saving content for {url}: {str(e)}")
        else:
            self.logger.warning(f"No significant content found for: {url}")

        for tag in main_content.find_all(['a', 'embed', 'iframe', 'object'], href=True):
            href = tag.get('href') or tag.get('src')
            if href:
                file_url = urljoin(url, href)
                if self.is_downloadable_file(file_url) and file_url not in self.downloaded_files:
                    self.download_file(file_url)

    def load_downloaded_files(self):
        """
        Loads the history of downloaded files from a tracking file.
        """
        downloaded_files_path = self.base_dir / self.config.output.logs_dir / 'downloaded_files.txt'
        if downloaded_files_path.exists():
            try:
                with open(downloaded_files_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        self.downloaded_files.add(line.strip())
                self.logger.info(f"Loaded {len(self.downloaded_files)} downloaded files.")
            except Exception as e:
                self.logger.error(f"Error loading downloaded files tracking: {str(e)}")
                self._update_metrics('errors', f"Error loading downloaded files tracking: {str(e)}")
        else:
            self.logger.info("No downloaded files tracking found, starting without history.")

    def save_downloaded_files(self):
        """
        Saves the history of downloaded files to a tracking file.
        """
        downloaded_files_path = self.base_dir / self.config.output.logs_dir / 'downloaded_files.txt'
        try:
            with open(downloaded_files_path, 'w', encoding='utf-8') as f:
                for url in sorted(self.downloaded_files):
                    f.write(url + '\n')
            self.logger.info(f"Saved {len(self.downloaded_files)} downloaded files.")
        except Exception as e:
            self.logger.error(f"Error saving downloaded files tracking: {str(e)}")
            self._update_metrics('errors', f"Error saving downloaded files tracking: {str(e)}")

    def generate_report(self, duration: float, error: Optional[str] = None):
        """
        Generates a JSON report of the operations performed.

        Args:
            duration (float): Total execution time in seconds.
            error (Optional[str], optional): Description of any critical error if present. Defaults to None.
        """
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
            "status": "Completed with errors" if error else "Completed successfully",
            "visited_pages": sorted(self.visited_pages),
            "downloaded_files": sorted(self.downloaded_files),
            "errors": metrics['errors'] if metrics['errors'] else None
        }

        json_report_path = self.base_dir / 'report.json'
        try:
            with open(json_report_path, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=4, ensure_ascii=False)
            self.logger.info(f"JSON report generated successfully: {json_report_path}")
        except Exception as e:
            self.logger.error(f"Error generating JSON report: {str(e)}")
            self._update_metrics('errors', f"Error generating JSON report: {str(e)}")

    def generate_xml_sitemap(self):
        """
        Generates an XML sitemap of the crawled URLs.
        """
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
            self.logger.info(f"XML sitemap generated successfully: {xml_path}")
        except Exception as e:
            self.logger.error(f"Error generating XML sitemap: {str(e)}")
            self._update_metrics('errors', f"Error generating XML sitemap: {str(e)}")

    def save_checkpoint(self):
        """
        Saves the current state of the crawler for later resumption.
        """
        checkpoint_data = {
            "visited_pages": list(self.visited_pages),
            "downloaded_files": list(self.downloaded_files),
            "site_map": {k: list(v) for k, v in self.site_map.items()},
            "stats": dict(self.stats)
        }
        try:
            with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, ensure_ascii=False, indent=4)
            self.logger.info(f"Checkpoint saved: {self.checkpoint_file}")
        except Exception as e:
            self.logger.error(f"Error saving checkpoint: {str(e)}")
            self._update_metrics('errors', f"Error saving checkpoint: {str(e)}")

    def load_checkpoint(self):
        """
        Loads a previous state of the crawler to resume crawling.
        """
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                    checkpoint_data = json.load(f)
                self.visited_pages = set(checkpoint_data.get("visited_pages", []))
                self.downloaded_files = set(checkpoint_data.get("downloaded_files", []))
                self.site_map = {k: set(v) for k, v in checkpoint_data.get("site_map", {}).items()}
                self.stats = defaultdict(int, checkpoint_data.get("stats", {}))
                self.logger.info(f"Checkpoint loaded: {self.checkpoint_file}")
            except Exception as e:
                self.logger.error(f"Error loading checkpoint: {str(e)}")
                self._update_metrics('errors', f"Error loading checkpoint: {str(e)}")

    def crawl(self):
        """
        Starts the crawling process in three phases:
            1. URL extraction.
            2. Content extraction.
            3. Content rewriting (if LLM enabled).
        Also generates a report and a sitemap at the end of execution.
        """
        start_time = time.time()
        self.logger.info(f"Starting crawl of {self.start_url}")

        self.load_downloaded_files()
        self.load_checkpoint()
        error = None
        try:
            # Phase 1: URL Extraction
            self.logger.info("Phase 1: Starting URL extraction")
            if not self.visited_pages:
                # No checkpoint, start extraction
                self.extract_urls(self.start_url)
                self.save_checkpoint()
            else:
                self.logger.info("Checkpoint found, skipping URL extraction phase.")

            # Phase 2: Content Extraction
            self.logger.info("Phase 2: Starting content extraction")
            total_pages = len(self.visited_pages)
            if total_pages == 0:
                self.logger.info("No pages to process.")
            else:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    "[progress.percentage]{task.percentage:>3.0f}%",
                    TimeRemainingColumn(),
                    console=self.console
                ) as progress:
                    task = progress.add_task("Extracting content", total=total_pages)

                    with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
                        futures = {executor.submit(self.extract_content, url): url for url in self.visited_pages}
                        for future in as_completed(futures):
                            url = futures[future]
                            try:
                                future.result()
                                progress.advance(task)
                            except Exception as e:
                                self.logger.error(f"Error extracting {url}: {str(e)}")
                                self._update_metrics('errors', f"Error extracting {url}: {str(e)}")
                                progress.advance(task)
                self.logger.info("Phase 2: Content extraction completed.")

            self.save_checkpoint()

            # Phase 3: Content Rewriting
            if self.llm_enabled:
                self.logger.info("Phase 3: Starting content rewriting")
                self.content_rewriter.rewrite_all_contents()

            # Generate Report
            end_time = time.time()
            duration = end_time - start_time
            self.generate_report(duration, error=error)
            self.generate_xml_sitemap()

        except Exception as e:
            error = str(e)
            self.logger.error(f"Critical error during crawl: {str(e)}")
            self._update_metrics('errors', f"Critical error during crawl: {str(e)}")

        self.save_downloaded_files()
        if self.use_playwright and self.page:
            self.page.close()
            self.browser.close()
            self.playwright.stop()

    def stop_crawl(self):
        """
        Stops the crawling process by signaling an stop event.
        """
        self.logger.info("Stop signal received. Stopping crawler...")
        self.stop_event.set()

    def _update_metrics(self, key: str, value):
        """
        Updates global metrics in a thread-safe manner.

        Args:
            key (str): The metric key to update.
            value: The value to add or increment.
        """
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
    """
    Loads the configuration from a JSON file.

    Args:
        config_path (str, optional): Path to the configuration file. Defaults to 'config.json'.

    Returns:
        dict: Configuration dictionary.

    Raises:
        FileNotFoundError: If the configuration file is not found.
        ValueError: If the JSON file is malformed.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file {config_path} not found.")
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        return config_data
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON decoding error in {config_path}: {e}")


def main():
    """
    Main entry point of the script. Loads the configuration, initializes the crawler,
    and starts the crawling process. Also handles user interruptions.
    """
    try:
        config_data = load_config()
        config = Config.parse_obj(config_data)
    except (FileNotFoundError, ValueError, ValidationError) as e:
        print(f"Configuration error: {e}")
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
        crawler.logger.info("Crawler stopped by user.")


if __name__ == "__main__":
    main()
