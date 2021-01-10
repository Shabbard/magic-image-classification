import json, requests
from types import SimpleNamespace

base_url = "https://api.scryfall.com"

def GetSetCardCount(set_name):
    set_url = base_url + "/sets/" + set_name
    response = requests.get(set_url)
    current_set = json.loads(response.text, object_hook=lambda d: SimpleNamespace(**d))
    return current_set.card_count

def GrabCardData(set_name, card_number):
    card_url = base_url + "/cards/" + set_name + '/' + str(card_number)
    response = requests.get(card_url)

    return json.loads(response.text, object_hook=lambda d: SimpleNamespace(**d))

def GenerateLUT():
    data = []
    set_name = "2xm"

    cards_in_set = GetSetCardCount(set_name)

    for card_number in (number+1 for number in range(cards_in_set)):
        current_card = GrabCardData(set_name, card_number)
        if current_card.name not in data:
            data.append(current_card.name)

    with open('lut.txt', 'w') as f:
        for item in data:
            f.write("%s\n" % item)


if __name__ == '__main__':
    GenerateLUT()