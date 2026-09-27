from gaia_agent.tools import web_search


def test_web_search_returns_results_text():
    result = web_search("Python programming language")
    assert isinstance(result, str)
    assert len(result) > 0
    assert "python" in result.lower()


from gaia_agent.tools import fetch_page


def test_fetch_page_extracts_text():
    result = fetch_page("https://example.com")
    assert isinstance(result, str)
    assert "Example Domain" in result
