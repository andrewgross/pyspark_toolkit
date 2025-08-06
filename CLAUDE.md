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

Tests should be readable narratives:

def test_when_db_has_records_and_api_is_called_then_returns_expected_response():
    # when I have a database with records
    insert_records_into_db([...])

    # and I make an API call to fetch my endpoint
    response = client.get("/some/endpoint")

    # then I should get a 200 status code and the expected response
    assert response.status_code == 200
    assert response.json() == {"key": "value"}

Use the pattern:

- when ... (setup context)
- and ... (perform action)
- then ... (assert outcome)

Test names should reflect behavior clearly — what you're testing, under what condition, and what the expected outcome is.

### Fixtures

Fixtures are used to setup the test environment. They are defined in the `conftest.py` file.

Fixtures should be:

- Fast — avoid heavy setup.
- Reused — avoid creating new instances for each test.
- Isolated — avoid side effects between tests.


### Emoji

Never use emoji when responding, writing logging statements, or writing documentation.
