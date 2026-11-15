from mcp_servers.notes.server import create_notes_server


def main() -> None:
    server = create_notes_server()
    server.run()


if __name__ == "__main__":
    main()
