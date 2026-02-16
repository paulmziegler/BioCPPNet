import click


@click.command()
def main():
    """Simple entry point."""
    click.echo("Hello from the Dockerized Python Project!")


if __name__ == "__main__":
    main()
