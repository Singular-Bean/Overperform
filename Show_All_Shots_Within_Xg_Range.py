import requests
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tkinter as tk
from tkinter import simpledialog

def check_website(url):
    try:
        response = requests.get(url)
        # Check if the response status code is 200 (OK)
        if response.status_code == 200:
            return True
        else:
            return False
    except requests.RequestException as e:
        # Handle any exceptions (like network errors)
        print(f"Error checking {url}: {e}")
        return False

def add_website_if_valid(url, website_list):
    if check_website(url):
        website_list.append(url)



def fetch_and_parse_json(url):
    response = requests.get(url)
    response.raise_for_status(
    )  # Ensure we raise an error for bad status codes
    data = response.json()
    return data


def list_xg_from_shotmap(json_data, isHome):
    return [item['xg'] for item in json_data['shotmap'] if
            'xg' in item and item['isHome'] == isHome and item["situation"] != "shootout"]

def leagueid():
    league = input("What league would you like to view the shot xgs of? ")
    leagueid = fetch_and_parse_json("http://www.sofascore.com/api/v1/search/unique-tournaments?q=" + league + "&page=0")['results'][0]['entity']['id']
    return leagueid
def seasonid(leagueid):
    options = []
    src = []
    seasons = fetch_and_parse_json("http://www.sofascore.com/api/v1/unique-tournament/" + str(leagueid) + "/seasons")['seasons']
    for t in range (0,len(seasons)):
        add_website_if_valid("http://www.sofascore.com/api/v1/unique-tournament/" + str(leagueid) + "/season/" + str(seasons[t]['id']) + "/events/round/1", src)
    for i in range (0,len(src)):
        id = seasons[i]['id']
        if 'hasXg' in fetch_and_parse_json("http://www.sofascore.com/api/v1/unique-tournament/" + str(leagueid) + "/season/" + str(id) + "/events/round/1")['events'][0]:
            options.append(seasons[i]['year'])
            print(str(i+1) + ". " + options[len(options)-1])
    year = int(input("Which season number would you like to view the true table of? "))
    for l in range(0, len(seasons)):
        if seasons[l]['year'] == options[year-1]:
            return seasons[l]['id']

def teamid(name):
    url = "http://www.sofascore.com/api/v1/search/teams?q=" + name + "&page=0"
    data = fetch_and_parse_json(url)['results'][0]['entity']['id']
    return data

def match_list(seasonid, leagueid, round):
    full_list = []
    for p in range(0, int(round)):
        roundmatches = fetch_and_parse_json("http://www.sofascore.com/api/v1/unique-tournament/" + str(leagueid) + "/season/" + str(seasonid) + "/events/round/" + str(p+1))["events"]
        for i in range(0, len(roundmatches)):
            match = roundmatches[i]
            matchid = match["id"]
            full_list.append(matchid)
    return full_list


def roundcalc(leagueid, seasonid):
    teamnum = len(fetch_and_parse_json("http://www.sofascore.com/api/v1/unique-tournament/" + str(leagueid) + "/season/" + str(seasonid) + "/standings/total")['standings'][0]['rows'])
    return (teamnum - 1) * 2

def shotmap_data(matchid):
    event_ = fetch_and_parse_json("http://www.sofascore.com/api/v1/event/" + str(matchid))['event']
    if event_.get('status') != None:
        if event_['status']['type'] != 'postponed' and check_website("http://www.sofascore.com/api/v1/event/" + str(matchid) + "/shotmap"):
            return fetch_and_parse_json("http://www.sofascore.com/api/v1/event/" + str(matchid) + "/shotmap")['shotmap']
    elif check_website("http://www.sofascore.com/api/v1/event/" + str(matchid) + "/shotmap"):
        return fetch_and_parse_json("http://www.sofascore.com/api/v1/event/" + str(matchid) + "/shotmap")['shotmap']

def nonround(number, places):
    if number != None:
        return round(number, places)
    else:
        return None

def list_to_end_all_lists(matches, rndnum):
    biglist = []
    for i in range(0, len(matches)):
        if i / rndnum == int(i / rndnum):
            print("Loading...")
        shotmap = shotmap_data(matches[i])
        if shotmap != None:
            for j in range(0, len(shotmap)):
                item = shotmap[j]
                biglist.append((nonround(item.get('xg'), 2), nonround(item.get('xgot'), 2), item.get('playerCoordinates'), item.get('goalMouthCoordinates'), item.get('draw'), item.get('blockCoordinates'), item.get('shotType')))
    return biglist

lgid = leagueid()
ssid = seasonid(lgid)
rndnum = roundcalc(lgid, ssid)
matches = match_list(ssid, lgid, rndnum)



# Function to draw the pitch (same as before)
def draw_pitch():
    fig, ax = plt.subplots(figsize=(10, 7.5))

    # Set pitch dimensions
    pitch_length = 100  # y-axis (from 0 to 100)
    pitch_width = 75  # x-axis (from 0 to 52)

    # Add green stripes to the pitch
    stripe_width = pitch_width / 4
    for i in range(4):
        ax.add_patch(patches.Rectangle((0, i * stripe_width), pitch_length, stripe_width,
                                       color='#446C46' if i % 2 == 0 else '#537855', zorder=0))

    # Pitch Outline & Centre Line (for the half-pitch)
    plt.plot([0, 0], [0, 75], color="black", linewidth=4)
    plt.plot([0, 100], [75, 75], color="black", linewidth=4)
    plt.plot([100, 100], [75, 0], color="black", linewidth=4)
    plt.plot([100, 0], [0, 0], color="black", linewidth=4)

    # Penalty Area
    plt.plot([20, 20], [50, 75], color="black", linewidth=4)
    plt.plot([80, 80], [50, 75], color="black", linewidth=4)
    plt.plot([20, 80], [50, 50], color="black", linewidth=4)

    # 6-yard Box
    plt.plot([36.5, 36.5], [67, 75], color="black", linewidth=4)
    plt.plot([63.5, 63.5], [67, 75], color="black", linewidth=4)
    plt.plot([36.5, 63.5], [67, 67], color="black", linewidth=4)

    # Goal
    plt.plot([42.5, 57.5], [74, 74], color="black", linewidth=8, alpha=0.6)

    # Penalty Spot and Centre Spot
    penSpot = plt.Circle((50, 59), 0.5, color="black")
    ax.add_patch(penSpot)

    return fig, ax


# Function to plot shots based on selected xG value
def plot_shots(xg_min, xg_max):
    fig, ax = draw_pitch()
    goalcount = 0
    misscount = 0

    for shot in shots:
        xg, xgot, player_coords, shot_coords, draw_info, block_coords, shot_type = shot

        # Filter based on xG range
        if xg != None:
            if xg_min <= nonround(xg, 2) <= xg_max:
                shot_x = 1.4423*player_coords['x']
                shot_y = player_coords['y']

                startdrawy = draw_info['start']['x']
                startdrawx = 1.4423*draw_info['start']['y']
                if draw_info.get('block') != None:
                    blockdrawy = draw_info['block']['x']
                    blockdrawx = 1.4423*draw_info['block']['y']
                enddrawy = draw_info['end']['x']
                enddrawx = 1.4423*draw_info['end']['y']


                # Ensure shot_x and shot_y are in the correct range
                if 0 <= shot_x <= 75 and 0 <= shot_y <= 100 and shot_type == 'goal':
                    goalcount += 1
                    # Plot the shot, with shot_x becoming y-axis and shot_y becoming x-axis
                    ax.scatter(shot_y, 75 - shot_x, color='red', s=100, alpha=0.6, edgecolor='black')
                    ax.plot([startdrawy, enddrawy], [75 - startdrawx, 75 - enddrawx], color='red', linewidth=2, alpha=0.0)
                elif 0 <= shot_x <= 75 and 0 <= shot_y <= 100 and draw_info.get('block') != None:
                    misscount += 1
                    # Plot the shot, with shot_x becoming y-axis and shot_y becoming x-axis
                    ax.scatter(shot_y, 75 - shot_x, color='white', s=100, alpha=0.6, edgecolor='black')
                    ax.plot([startdrawy, blockdrawy], [75 - startdrawx, 75 - blockdrawx], color='white', linewidth=1, alpha=0.0)
                elif 0 <= shot_x <= 75 and 0 <= shot_y <= 100:
                    misscount += 1
                    # Plot the shot, with shot_x becoming y-axis and shot_y becoming x-axis
                    ax.scatter(shot_y, 75 - shot_x, color='white', s=100, alpha=0.6, edgecolor='black')
                    ax.plot([startdrawy, enddrawy], [75 - startdrawx, 75 - enddrawx], color='white', linewidth=1, alpha=0.0)

    plt.xlim(-2, 102)  # Matches the range of the half-pitch (length)
    plt.ylim(-2, 77)  # Matches the range of the half-pitch (width)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.text(50, -7.5, "Goals scored = " + str(goalcount) + "      Goals Missed = " + str(misscount) + "       Score rate = " + str(round(goalcount/(goalcount+misscount), 4)) + "%", ha='center', fontsize=12, color='black')
    plt.show()


# Tkinter dialog to get user input for xG value
def get_xg_range_and_plot():
    root = tk.Tk()
    root.withdraw()  # Hide the main Tkinter window

    # Prompt the user to enter the minimum xG value
    xg_min = simpledialog.askfloat("Input", "Enter minimum xG value (0.00 - 0.99):", minvalue=0.00, maxvalue=0.99)

    # Prompt the user to enter the maximum xG value
    xg_max = simpledialog.askfloat("Input", "Enter maximum xG value (0.00 - 0.99):", minvalue=0.00, maxvalue=0.99)

    # If xg_max is None, assume they wanted a single value range
    if xg_max is None:
        xg_max = xg_min

    # Plot the shots based on the user input
    if xg_min is not None and xg_max is not None:
        plot_shots(xg_min, xg_max)


# Example shot data (same as before)
shots = list_to_end_all_lists(matches, rndnum)

# Run the Tkinter dialog and plot
get_xg_range_and_plot()