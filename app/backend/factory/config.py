import os
from dotenv import load_dotenv

load_dotenv()


class Config(object):
    SQLALCHEMY_DATABASE_URI = os.getenv("DATABASE_URl")
    SECRET_KEY = os.getenv("SECRET_KEY")
    SESSION_COOKIE_NAME = "session"
    MODEL = "test"
    SESSION_PERMANENT = True
    LOGDIR = "app/logs/"
    FASTTEXT = "app/backend/core/embed/fasttext/model.model"
    CASSANDRA_HOST = os.getenv("CASSANDRA_HOST")
    CASSANDRA_KEYSPACE = os.getenv("CASSANDRA_KEYSPACE")
    CASSANDRA_LAZY_CONNECT = True
