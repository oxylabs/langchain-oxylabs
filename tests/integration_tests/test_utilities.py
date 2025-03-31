"""Integration test for OxylabsSearchAPIWrapper."""
from langchain_oxylabs import OxylabsSearchAPIWrapper

def test_call() -> None:
    """Test that call gives the correct answer."""
    chain = OxylabsSearchAPIWrapper()
    output = chain.run("Python programming language")
    assert "high-level, general-purpose programming language" in output
    assert ".py" in output
    assert "Guido van Rossum" in output

async def test_async_call() -> None:
    """Test that call gives the correct answer."""
    chain = OxylabsSearchAPIWrapper()
    output = chain.run("Python programming language")
    assert "high-level, general-purpose programming language" in output
    assert ".py" in output
    assert "Guido van Rossum" in output