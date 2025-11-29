from base.constants import Constants
from ddgs import DDGS
from trafilatura import extract

import requests
import time

class WebSearch:
    """
    A simple web search class that uses DuckDuckGo to fetch answers.
    """
    def _collect_urls(self, query: str, max_results: int = 3):
        """Collects URLs from DuckDuckGo search results.

        Args:
            query (str): The search query.
            max_results (int): The maximum number of URLs to collect.

        Returns:
            list: A list of URLs from the search results.
        """
        urls = []
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results = max_results)
            for result in results:
                url = result.get('href', None)
                if url:
                    urls.append(url)
        return urls
    

    def _fetch_page_content(self, url: str, user_agent_header: str = Constants.URLHeader.USER_AGENT_HEADER, timeout: int = Constants.Timeout.TEN_SECONDS) -> str:
        """ Fetches and extracts text content from a web page.
        Args:
            url (str): The URL of the web page to fetch.
            user_agent_header (str): The User-Agent header to use for the HTTP request.
            timeout (int): The timeout for the HTTP request in seconds.

        Returns:
            str: The extracted text content from the web page, or None if fetching fails.
        """
        headers = {'User-Agent': user_agent_header}
        try:
            response = requests.get(url, headers = headers, timeout = timeout)
            if response.status_code == Constants.HTTPStatusCodes.OK:
                html = response.text
                text = extract(html)
                return text
            raise Exception(f"Failed to fetch {url}; status code:{response.status_code}")
        except Exception as e:
            raise Exception(f"Error fetching URL {url}: {str(e)}")
        

    def search(self, query: str, max_results: int = 3, timeout: int = Constants.Timeout.THREE_SECONDS) -> list:
        """Performs a web search and retrieves content from the top results.

        Args:
            query (str): The search query.
            max_results (int): The maximum number of search results to consider.
            timeout (int): The timeout (in seconds) between fetching each web page

        Returns:
            list: Collected contents from the web pages. Returns a list with a placeholder if no content is found.
        """
        urls = self._collect_urls(query, max_results)
        documents = []
        for url in urls:
            try:
                content = self._fetch_page_content(url)
                if content:
                    documents.append({'url': url, 'content': content})
                time.sleep(timeout)
            except:
                pass
        if not documents:
            return [{'url': "NA", 'content': "NA"}]
        return documents
        

    