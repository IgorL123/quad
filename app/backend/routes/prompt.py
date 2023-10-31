from flask import (
    request,
    Blueprint,
    redirect,
    url_for,
    make_response,
    flash,
    current_app,
)
from flask_login import LoginManager, login_required, current_user
from ..models import db, Request, Response
from ..core import main

prompt = Blueprint("prompt", __name__)
login_manager = LoginManager()
login_manager.init_app(prompt)


@prompt.route("/prompt", methods=["GET", "POST"])
@login_required
def show():
    if request.method == "POST":
        text = request.form["text"]

        new_req = Request(text=text, id_user=current_user.id)
        db.session.add(new_req)
        db.session.commit()
        current_app.logger.info(f"New prompt from user with id {current_user.id}")

        current_app.logger.info("Pipeline start running...")
        res = main.query(text)
        current_app.logger.info("Pipeline end running")

        if len(res) >= 1000:
            text = res[:1000]
        else:
            text = res
        new_res = Response(
            text=text,
            grade=0,
            id_request=new_req.id,
            model_type=current_app.config["MODEL"],
        )
        db.session.add(new_res)
        db.session.commit()

        id_response = str(new_res.id)

        return make_response(
            redirect(
                url_for(
                    "home.show",
                    result=text,
                    id_response=id_response,
                )
            )
        )

    if request.method == "GET":
        return redirect(url_for("/home.show"))
