## Writing Python Tests (PyTest)

We use PyTest for all Python testing. Tests should be:

- Fast and reliable — no flaky tests.
- Never commented out due to failure — fix them or ask for help.
- Simple to write — good code design should make tests easy to write without excessive mocking

### Guidelines

- Use test functions or classes (with PyTest).
- Use fixtures only when helpful.
- Avoid mocks and stubs when possible — prefer passing real inputs and inspecting real outputs.
- Tests should not need heavy setup if the code is clean and modular.
- Tests should be able to run in parallel and in any order. We should not rely on the order of tests to run.

### Test Structure

Tests should be readable narratives with clear comments marking each section:

def test_when_db_has_records_and_api_is_called_then_returns_expected_response():
    """
    Test that API returns expected response when database has records.
    This ensures our endpoint correctly retrieves and formats data.
    """
    # when I have a database with records
    insert_records_into_db([...])

    # and I make an API call to fetch my endpoint
    response = client.get("/some/endpoint")

    # then I should get a 200 status code and the expected response
    assert response.status_code == 200
    assert response.json() == {"key": "value"}

Use the pattern:

- Docstring: Explain what the test is doing and why (if needed)
- Comments: Use `# when`, `# and`, `# then` to mark test sections
- Test names should reflect behavior clearly — what you're testing, under what condition, and what the expected outcome is

### Test Focus

Each test should focus on testing only one action or concern:

- A single test can have multiple assertions to verify the complete state
- But avoid testing multiple unrelated behaviors in one test
- If you find yourself testing two different concerns, split into separate tests

### Fixtures

Fixtures are used to setup the test environment. They are defined in the `conftest.py` file.

Fixtures should be:

- Fast — avoid heavy setup.
- Reused — avoid creating new instances for each test.
- Isolated — avoid side effects between tests.


### Emoji

Never use emoji when responding, writing logging statements, or writing documentation.
