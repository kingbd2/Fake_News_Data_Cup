import requests
import json

def SlackNotification(channel, message):
    if channel == "BK_slackbot":
        url = 'https://hooks.slack.com/services/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'
    elif channel == "datacup":
        url = 'https://hooks.slack.com/services/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'
    else:
        return("Could not find the channel")
    data = {"text": str(message)}
    headers = {"Content-Type": "application/json"}
    response = requests.post(
        url, data=json.dumps(data),
        headers=headers
    )
    if response.status_code != 200:
        raise ValueError(
            'Request to slack returned an error %s, the response is:\n%s'
            % (response.status_code, response.text)
        )