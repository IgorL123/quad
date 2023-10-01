import pytest
from app.backend.factory import create_app
import os


@pytest.fixture
def client():
    os.environ["CONFIG_TYPE"] = "config.TestingConfig"
    app = create_app()
    with app.test_client() as client:
        with app.app_context():
            yield client


def test_home_page(client):
    response = client.get("/home")
    assert response.status_code == 200


def test_login_page(client):
    response = client.get("/login")
    assert response.status_code == 200
    assert b"Login" in response.data


def test_register_page(client):
    response = client.get("/register")
    assert response.status_code == 200
    assert b"Register" in response.data
