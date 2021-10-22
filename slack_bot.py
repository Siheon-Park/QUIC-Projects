import logging
from logging.handlers import HTTPHandler
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

SLACK_TOKEN = "xoxb-2434356654884-2440936999633-fOUgZEZREvXWpyVvGhyxjtT8"
CHANNELS = {
    'DEBUG': 'C02DHK2DQBY',
    'INFO': 'C02DWHNH71A',
    'WARNING': 'C02DMB5EDLM',
    'ERROR': 'C02DQBR7MEW',
    'CRITICAL': 'C02DWHTJZSQ'
}
MYUSERID = 'U02CH3W4Y6R'

logger = logging.getLogger(__name__)


class SlackHandler(logging.handlers.HTTPHandler):
    def __init__(self, token=SLACK_TOKEN, emoji=True):
        super().__init__(host='slack.com', url='/api/chat.postMessage', secure=True, method='POST')
        self.token = token
        self.emoji = emoji

    def mapLogRecord(self, record):
        if self.formatter is None:  # Formatter가 설정되지 않은 경우
            text = record.msg
        else:
            text = self.formatter.format(record)

        emoji = (
            '' if self.emoji == False else
            ':bug:' if record.levelname == 'DEBUG' else
            ':pencil2:' if record.levelname == 'INFO' else
            ':warning:' if record.levelname == 'WARNING' else
            ':no_entry:' if record.levelname == 'ERROR' else
            ':rotating_light:' if record.levelname == 'CRITICAL' else
            ''
        )

        return {
            'token': self.token,
            'channel': CHANNELS[record.levelname],
            'text': f'{emoji} {record.levelname} {text}',
            'as_user': True,
        }


class SlackWebClient(WebClient):
    def __init__(self, token=SLACK_TOKEN, *args, **kwargs):
        super(SlackWebClient, self).__init__(token=token, *args, **kwargs)

    def post_file(self, file_name, text: str = None, channels=CHANNELS["INFO"], mention=False):
        try:
            # Call the files.upload method using the WebClient
            # Uploading files requires the `files:write` scope
            initial_comment = (
                None if text is None and not mention else
                text if text is not None and not mention else
                f"<@{MYUSERID}>" if text is None and mention else
                f"<@{MYUSERID}> {text}"  # if text is not None and mention
            )
            result = self.files_upload(
                channels=channels,
                initial_comment=initial_comment,
                file=str(file_name),
                filename=str(file_name)
            )
            # Log the result
            logger.info(result)

        except SlackApiError as e:
            logger.error("Error uploading file: {}".format(e))

    def post_message(self, text: str, channel=CHANNELS["INFO"], mention=False):
        try:
            if mention:
                text = f'<@{MYUSERID}> {text}'
            # Call the chat.postMessage method using the WebClient
            result = self.chat_postMessage(
                channel=channel,
                text=text,
                as_user=True,
            )
            logger.info(result)

        except SlackApiError as e:
            logger.error(f"Error posting message: {e}")
