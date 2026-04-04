"""PubMed / NCBI E-utilities client for paper and dataset discovery."""
from __future__ import annotations

import xml.etree.ElementTree as ET

import httpx
import structlog
from pydantic import BaseModel, Field

log = structlog.get_logger()

_EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
_TIMEOUT = 20.0
_MAX_RESULTS = 20


class PubMedArticle(BaseModel):
    """A PubMed article summary."""
    pmid: str
    title: str = ""
    abstract: str = ""
    authors: list[str] = Field(default_factory=list)
    journal: str = ""
    pub_date: str = ""
    doi: str = ""
    pmc_id: str = ""  # PMC ID if available (open access)

    @property
    def url(self) -> str:
        return f"https://pubmed.ncbi.nlm.nih.gov/{self.pmid}/"

    @property
    def has_full_text(self) -> bool:
        return bool(self.pmc_id)


class GEODataset(BaseModel):
    """A GEO dataset linked to a PubMed article."""
    accession: str  # e.g. GSE12345
    title: str = ""
    pmid: str = ""

    @property
    def url(self) -> str:
        return f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={self.accession}"


async def search(
    query: str,
    *,
    max_results: int = _MAX_RESULTS,
    api_key: str = "",
) -> list[str]:
    """Search PubMed and return a list of PMIDs."""
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": str(max_results),
        "retmode": "json",
        "sort": "relevance",
    }
    if api_key:
        params["api_key"] = api_key

    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.get(f"{_EUTILS_BASE}/esearch.fcgi", params=params)
            resp.raise_for_status()

        data = resp.json()
        pmids = data.get("esearchresult", {}).get("idlist", [])
        log.info("pubmed.search", query=query[:50], results=len(pmids))
        return pmids

    except Exception:
        log.exception("pubmed.search_error", query=query[:50])
        return []


async def fetch_articles(
    pmids: list[str],
    *,
    api_key: str = "",
) -> list[PubMedArticle]:
    """Fetch article details for a list of PMIDs."""
    if not pmids:
        return []

    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "rettype": "xml",
        "retmode": "xml",
    }
    if api_key:
        params["api_key"] = api_key

    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.get(f"{_EUTILS_BASE}/efetch.fcgi", params=params)
            resp.raise_for_status()

        return _parse_pubmed_xml(resp.text)

    except Exception:
        log.exception("pubmed.fetch_error", pmids=pmids[:5])
        return []


async def find_geo_datasets(
    pmid: str,
    *,
    api_key: str = "",
) -> list[GEODataset]:
    """Find GEO datasets linked to a PubMed article via ELink."""
    params = {
        "dbfrom": "pubmed",
        "db": "gds",
        "id": pmid,
        "retmode": "json",
    }
    if api_key:
        params["api_key"] = api_key

    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.get(f"{_EUTILS_BASE}/elink.fcgi", params=params)
            resp.raise_for_status()

        data = resp.json()
        linksets = data.get("linksets", [])
        datasets = []

        for linkset in linksets:
            for linkdb in linkset.get("linksetdbs", []):
                if linkdb.get("dbto") == "gds":
                    for link in linkdb.get("links", []):
                        datasets.append(
                            GEODataset(accession=f"GDS{link}", pmid=pmid)
                        )

        log.info("pubmed.geo_links", pmid=pmid, datasets=len(datasets))
        return datasets

    except Exception:
        log.exception("pubmed.geo_error", pmid=pmid)
        return []


async def search_and_fetch(
    query: str,
    *,
    max_results: int = 10,
    api_key: str = "",
) -> list[PubMedArticle]:
    """Search PubMed and return full article details."""
    pmids = await search(query, max_results=max_results, api_key=api_key)
    if not pmids:
        return []
    return await fetch_articles(pmids, api_key=api_key)


def _parse_pubmed_xml(xml_text: str) -> list[PubMedArticle]:
    """Parse PubMed XML response into article models."""
    articles = []

    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        log.warning("pubmed.xml_parse_error")
        return []

    for article_el in root.findall(".//PubmedArticle"):
        medline = article_el.find(".//MedlineCitation")
        if medline is None:
            continue

        pmid_el = medline.find("PMID")
        pmid = pmid_el.text if pmid_el is not None else ""

        art = medline.find(".//Article")
        if art is None:
            continue

        title_el = art.find("ArticleTitle")
        title = title_el.text if title_el is not None else ""

        # Abstract can have multiple AbstractText elements
        abstract_parts = []
        for abs_el in art.findall(".//AbstractText"):
            label = abs_el.get("Label", "")
            text = abs_el.text or ""
            if label:
                abstract_parts.append(f"{label}: {text}")
            else:
                abstract_parts.append(text)
        abstract = "\n".join(abstract_parts)

        # Authors
        authors = []
        for author_el in art.findall(".//Author"):
            last = author_el.findtext("LastName", "")
            initials = author_el.findtext("Initials", "")
            if last:
                authors.append(f"{last} {initials}".strip())

        journal_el = art.find(".//Journal/Title")
        journal = journal_el.text if journal_el is not None else ""

        # DOI
        doi = ""
        for id_el in article_el.findall(".//ArticleId"):
            if id_el.get("IdType") == "doi":
                doi = id_el.text or ""
                break

        # PMC ID
        pmc_id = ""
        for id_el in article_el.findall(".//ArticleId"):
            if id_el.get("IdType") == "pmc":
                pmc_id = id_el.text or ""
                break

        # Publication date
        pub_date = ""
        date_el = art.find(".//Journal/JournalIssue/PubDate")
        if date_el is not None:
            year = date_el.findtext("Year", "")
            month = date_el.findtext("Month", "")
            pub_date = f"{year} {month}".strip()

        articles.append(
            PubMedArticle(
                pmid=pmid,
                title=title,
                abstract=abstract,
                authors=authors,
                journal=journal,
                pub_date=pub_date,
                doi=doi,
                pmc_id=pmc_id,
            )
        )

    log.info("pubmed.parsed", articles=len(articles))
    return articles
