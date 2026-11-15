
from mcp_servers.calendar.client import GoogleCalendarClient
from mcp_servers.calendar.server import create_calendar_server


def main() -> None:
    client = GoogleCalendarClient()
    server = create_calendar_server(client)

    try:
        server.run()
    finally:
        client.close()


if __name__ == "__main__":
    main()
