from mcp_servers.github.server import create_github_server


def main() -> None:
    server = create_github_server()
    server.run()


if __name__ == "__main__":
    main()
