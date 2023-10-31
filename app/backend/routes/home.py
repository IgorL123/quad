from flask import Blueprint, render_template, request, current_app
from flask_login import LoginManager, login_required
from ..core import main

home = Blueprint("home", __name__)
login_manager = LoginManager()
login_manager.init_app(home)


@home.route("/home", methods=["GET"])
@login_required
def show():
    res = request.args.get("result")
    id_response = request.args.get("id_response")

    tmp = ["FredT5_saiga"]
    return render_template(
        "home_b.html",
        models=tmp,
        result=res,
        id_response=id_response,
        model_type=current_app.config["MODEL"],
    )
