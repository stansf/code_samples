from pprint import pprint

import timm
import typer

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)


@app.command()
def main(
        fltr: str = '',
        show_cfg: bool = False
):
    model_names = timm.list_models(f'*{fltr}*')
    pprint(model_names)
    if show_cfg:
        for model_name in model_names:
            model = timm.create_model(model_name)
            pprint(model.pretrained_cfg)


if __name__ == '__main__':
    app()
