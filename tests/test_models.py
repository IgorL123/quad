from app.backend.models import Users


def test_users():
    email = "test@test.mail"
    pasw = "TEST_PASSWORD"

    user = Users(email=email, password=pasw)
    assert user.email == email
    assert user.password_hash == pasw


