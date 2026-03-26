import asyncio
import ast
import time
from src.agents.LLMCompiler.tools.base import Tool
from typing import List, Optional, Union, Any, Optional, Type
import aiohttp
import requests
from bs4 import BeautifulSoup
from langchain_core.tools import BaseTool
from langchain_core.documents import Document
from langchain_community.docstore.base import Docstore
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper

def clean_str(p):
    try:
        return p.encode().decode("unicode-escape").encode("latin1").decode("utf-8")
    except Exception as e:
        print(e)
        return p

class Wikipedia(Docstore):
    """Wrapper around wikipedia API."""

    def __init__(self, benchmark=False, skip_retry_when_postprocess=False) -> None:
        """Check that wikipedia package is installed."""
        try:
            import requests
            from bs4 import BeautifulSoup
        except ImportError:
            raise ImportError(
                "Could not import wikipedia python package. "
                "Please install it with `pip install wikipedia`."
            )
        self.benchmark = benchmark
        self.all_times = []

        # when True, always skip retry when postprocess
        self.skip_retry_when_postprocess = skip_retry_when_postprocess

    def reset(self):
        self.all_times = []

    def get_stats(self):
        return {
            "all_times": self.all_times,
        }

    @staticmethod
    def _get_page_obs(page):
        # find all paragraphs
        paragraphs = page.split("\n")
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        # find all sentence
        sentences = []
        for p in paragraphs:
            sentences += p.split(". ")
        sentences = [s.strip() + "." for s in sentences if s.strip()]
        return " ".join(sentences[:5])

    def _get_alternative(self, result: str) -> str:
        parsed_alternatives = result.split("Similar: ")[1][:-1]

        alternatives = ast.literal_eval(parsed_alternatives)
        alternative = alternatives[0]
        for alt in alternatives:
            if "film" in alt or "movie" in alt:
                alternative = alt
                break
        return alternative

    def post_process(
        self, response_text: str, entity: str, skip_retry_when_postprocess: bool = False
    ) -> str:
        soup = BeautifulSoup(response_text, features="html.parser")
        result_divs = soup.find_all("div", {"class": "mw-search-result-heading"})

        if result_divs:  # mismatch
            self.result_titles = [
                clean_str(div.get_text().strip()) for div in result_divs
            ]
            obs = f"Could not find {entity}. Similar: {self.result_titles[:5]}."
        else:
            page = [
                p.get_text().strip() for p in soup.find_all("p") + soup.find_all("ul")
            ]
            if any("may refer to:" in p for p in page):
                if skip_retry_when_postprocess or self.skip_retry_when_postprocess:
                    obs = "Could not find " + entity + "."
                else:
                    obs = self.search("[" + entity + "]", is_retry=True)
            else:
                page_str = ""
                for p in page:
                    if len(p.split(" ")) > 2:
                        page_str += clean_str(p)
                        if not p.endswith("\n\n"):
                            page_str += "\n\n"
                # obs = self._get_page_obs(page_str)
                obs = page_str
                
        obs = obs.replace("\\n", "")
        return obs

    async def apost_process(
        self, response_text: str, entity: str, skip_retry_when_postprocess: bool = False
    ) -> str:
        soup = BeautifulSoup(response_text, features="html.parser")
        result_divs = soup.find_all("div", {"class": "mw-search-result-heading"})

        if result_divs:  # mismatch
            self.result_titles = [
                clean_str(div.get_text().strip()) for div in result_divs
            ]
            obs = f"Could not find {entity}. Similar: {self.result_titles[:5]}."
        else:
            page = [
                p.get_text().strip() for p in soup.find_all("p") + soup.find_all("ul")
            ]
            if any("may refer to:" in p for p in page):
                if skip_retry_when_postprocess or self.skip_retry_when_postprocess:
                    obs = "Could not find " + entity + "."
                else:
                    obs = await self.asearch("[" + entity + "]", is_retry=True)
            else:
                page_str = ""
                for p in page:
                    if len(p.split(" ")) > 2:
                        page_str += clean_str(p)
                        if not p.endswith("\n\n"):
                            page_str += "\n\n"
                # obs = self._get_page_obs(page_str)
                obs = page_str
        obs = obs.replace("\\n", "")
        return obs

    def search(self, entity: str, is_retry: bool = False) -> Union[str, Document]:
        """Try to search for wiki page.

        If page exists, return the page summary, and a PageWithLookups object.
        If page does not exist, return similar entries.

        Args:
            entity: entity string.

        Returns: a Document object or error message.
        """
        s = time.time()
        entity = str(entity)
        entity_ = entity.replace(" ", "+")
        search_url = f"https://en.wikipedia.org/w/index.php?search={entity_}"
        response_text = requests.get(search_url).text
        result = self.post_process(response_text, entity)

        if "Similar:" in result:
            alternative = self._get_alternative(result)
            entity_ = alternative.replace(" ", "+")
            search_url = f"https://en.wikipedia.org/w/index.php?search={entity_}"
            response_text = requests.get(search_url).text

            result = self.post_process(
                response_text, entity, skip_retry_when_postprocess=True
            )

            if "Similar:" in result:
                result = "Could not find " + entity + "."

        if self.benchmark and not is_retry:
            # we only benchmark the outermost call
            self.all_times.append(round(time.time() - s, 2))

        return result

    async def asearch(
        self, entity: str, is_retry: bool = False
    ) -> Union[str, Document]:
        """Try to search for wiki page.

        If page exists, return the page summary, and a PageWithLookups object.
        If page does not exist, return similar entries.

        Args:
            entity: entity string.

        Returns: a Document object or error message.
        """
        s = time.time()
        entity = str(entity)
        entity_ = entity.replace(" ", "+")
        search_url = f"https://en.wikipedia.org/w/index.php?search={entity_}"

        async with aiohttp.ClientSession() as session:
            async with session.get(search_url) as response:
                response_text = await response.text()

        result = await self.apost_process(response_text, entity)

        if "Similar:" in result:
            alternative = self._get_alternative(result)
            entity_ = alternative.replace(" ", "+")
            search_url = f"https://en.wikipedia.org/w/index.php?search={entity_}"
            async with aiohttp.ClientSession() as session:
                async with session.get(search_url) as response:
                    response_text = await response.text()

            result = await self.apost_process(
                response_text, entity, skip_retry_when_postprocess=True
            )

            if "Similar:" in result:
                return "Could not find " + entity + "."

        if self.benchmark and not is_retry:
            # we only benchmark the outermost call
            self.all_times.append(round(time.time() - s, 2))

        return Document(page_content=result)


class DocstoreExplorer:
    """Class to assist with exploration of a document store."""

    def __init__(self, top_k_result = 1, char_limit=None, one_sentence=False):
        """Initialize with a docstore, and set initial document to None."""
        self.docstore = WikipediaAPIWrapper(top_k_results=top_k_result)
        self.docstore_old = Wikipedia()
        self.documents: dict[str, Document] = {}
        self.lookup_strs: dict[str, str] = {}
        self.lookup_indices: dict[str, int] = {}
        self.lookup_index = 0
        self.char_limit = char_limit
        self.one_sentence = one_sentence

    def _normalize_document(self, result: Any) -> Document:
        if isinstance(result, Document):
            content = result.page_content
            metadata = dict(getattr(result, "metadata", {}) or {})
        else:
            content = str(result)
            metadata = {}

        content = content.strip()
        if self.one_sentence and content:
            content = content.split(". ", 1)[0].strip()
        if self.char_limit is not None:
            content = content[: self.char_limit]
        return Document(page_content=content, metadata=metadata)

    def _store_document(self, term: str, result: Any) -> str:
        document = self._normalize_document(result)
        key = term.lower()
        self.documents[key] = document

        title = str(document.metadata.get("title", "")).strip().lower()
        if title:
            self.documents[title] = document

        return self.get_summary(key)

    def search(self, term: str) -> str:
        """Search for a term in the docstore, and if found save."""
        result: Any = None
        try:
            loaded = self.docstore.load(term)
            if loaded:
                result = loaded[0]
        except Exception:
            result = None

        if result is None:
            result = self.docstore_old.search(term)

        return self._store_document(term, result)

    async def asearch(self, term: str) -> str:
        """Search for a term in the docstore, and if found save."""
        return await asyncio.to_thread(self.search, term)

    def _paragraph_matches(self, page: str, term: str) -> List[str]:
        paragraphs = self.get_paragraphs(page)
        needle = term.lower().strip()
        if not needle:
            return paragraphs

        exact = [p for p in paragraphs if needle in p.lower()]
        if exact:
            return exact

        tokens = [tok for tok in needle.split() if tok]
        if len(tokens) > 1:
            token_matches = [
                p for p in paragraphs if all(tok in p.lower() for tok in tokens)
            ]
            if token_matches:
                return token_matches
        return []

    def lookup(self, page: str, term: str) -> str:
        """Lookup a term in document (if saved)."""
        page = page.lower()
        term = term.lower()
        if page not in self.documents:
            raise ValueError("Cannot lookup without a successful search first")
        if page in self.lookup_strs:
            lookup_str = self.lookup_strs[page]
        else: 
            lookup_str = ""
        if term != lookup_str:
            self.lookup_strs[page] = term
            self.lookup_indices[page] = 0
        else:
            self.lookup_indices[page] += 1
        lookup_str = term
        lookups = self._paragraph_matches(page, lookup_str)
        if len(lookups) == 0:
            return "No Results"
        elif self.lookup_indices[page] >= len(lookups):
            return "No More Results"
        else:
            result_prefix = f"(Result {self.lookup_indices[page] + 1}/{len(lookups)})"
            return f"{result_prefix} {lookups[self.lookup_indices[page]]}"
        
    async def alookup(self, page: str, term: str) -> str:
        page = page.lower()
        term = term.lower()
        if page not in self.documents:
            raise ValueError(f"Cannot lookup without a successful search first, pages: {list(self.documents.keys())}")
        if page in self.lookup_strs:
            lookup_str = self.lookup_strs[page]
        else: 
            lookup_str = ""
        if term != lookup_str:
            self.lookup_strs[page] = term
            self.lookup_indices[page] = 0
        else:
            self.lookup_indices[page] += 1
        lookup_str = term
        lookups = self._paragraph_matches(page, lookup_str)
        if len(lookups) == 0:
            return "No Results"
        elif self.lookup_indices[page] >= len(lookups):
            return "No More Results"
        else:
            result_prefix = f"(Result {self.lookup_indices[page] + 1}/{len(lookups)})"
            return f"{result_prefix} {lookups[self.lookup_indices[page]]}"

    def get_summary(self, page: str) -> str:
        if page not in self.documents:
            raise ValueError("Cannot get paragraphs without a search")
        paragraphs = self.get_paragraphs(page)
        if paragraphs:
            return paragraphs[0]
        return self.documents[page].page_content.strip()

    def get_paragraphs(self, page: str) -> List[str]:
        if page not in self.documents:
            raise ValueError("Cannot get paragraphs without a search")
        content = self.documents[page].page_content
        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
        if paragraphs:
            return paragraphs
        stripped = content.strip()
        return [stripped] if stripped else []

docstore = DocstoreExplorer()

def clear_pages():
    global docstore
    docstore.documents = {}
    docstore.lookup_strs = {}
    docstore.lookup_indices = {}

class WikipediaTool(BaseTool):
    name: str = "search"
    description: str = """Search text on Wikipedia"""
    wikipedia: DocstoreExplorer = docstore
    
    def _run(self, text: str) -> str:
        return self.wikipedia.search(text)

    async def _arun(self, text: str) -> str:
        """Asynchronously run the search tool."""
        try:
            return await self.wikipedia.asearch(text)
        except Exception as e:
            return e

class LookupTool(BaseTool):
    name: str = "lookup"
    description: str = """Returns the next sentence containing keyword from the previous search result. You need to specify which page to lookup. The page should be one of the previous search query from trajectories."""
    docstore: DocstoreExplorer = docstore
    
    def _run(self, page: str, keyword: str) -> str:
        return self.docstore.lookup(page, keyword)

    async def _arun(self, page: str, keyword: str) -> str:
        """Asynchronously run the search tool."""
        try:
            return await self.docstore.alookup(page, keyword)
        except Exception as e:
            return e

search = WikipediaTool()
lookup = LookupTool()


def _coerce_search_text(payload: Any) -> str:
    if isinstance(payload, dict):
        return str(payload.get("text", "")).strip()
    return str(payload).strip()


def _coerce_lookup_payload(payload: Any) -> tuple[str, str]:
    if isinstance(payload, dict):
        page = str(payload.get("page", "")).strip()
        keyword = str(payload.get("keyword", "")).strip()
        return page, keyword
    if isinstance(payload, (list, tuple)) and len(payload) >= 2:
        return str(payload[0]).strip(), str(payload[1]).strip()
    raise ValueError(f"Invalid lookup payload: {payload!r}")


async def _search_tool(payload: Any) -> str:
    text = _coerce_search_text(payload)
    return await search._arun(text)


async def _lookup_tool(payload: Any) -> str:
    page, keyword = _coerce_lookup_payload(payload)
    return await lookup._arun(page, keyword)


def _stringify_search(args: Any) -> str:
    payload = args[0] if args else ""
    if isinstance(payload, dict):
        text = payload.get("text", "")
        return f'search({{"text": "{text}"}})'
    return f"search({payload})"


def _stringify_lookup(args: Any) -> str:
    payload = args[0] if args else {}
    if isinstance(payload, dict):
        page = payload.get("page", "")
        keyword = payload.get("keyword", "")
        return f'lookup({{"page": "{page}", "keyword": "{keyword}"}})'
    return f"lookup({payload})"

tools = [
    Tool(
        name="search",
        func=_search_tool,
        description=(
            'search({"text": "entity or topic"}) -> str:\n'
            " - Search Wikipedia for an entity or topic.\n"
            " - Use a concise page title or entity name.\n"
            " - Returns the loaded page summary/lead text."
        ),
        stringify_rule=_stringify_search,
    ),
    Tool(
        name="lookup",
        func=_lookup_tool,
        description=(
            'lookup({"page": "searched page title", "keyword": "keyword to scan for"}) -> str:\n'
            + lookup.description
            + " Use the lookup tool only after you have searched for the page."
        ),
        stringify_rule=_stringify_lookup,
    ),
]
