import matplotlib.pyplot as plt
import numpy as np
import requests
from collections import defaultdict
import matplotlib.patches as patches

def zerodivide(numerator, denominator):
    if denominator == 0:
        return 0
    else:
        return numerator/denominator

def find(x, y):
    if y in x:
        return x[y]
    else:
        return None

def nonround(number, places):
    if number != None:
        return round(number, places)
    else:
        return None


def probability_of_heads(probabilities, target_heads):
    """
    Calculate the probability of getting exactly `target_heads` heads from flipping each coin once.

    Args:
        probabilities (list): A list of probabilities for getting heads for each biased coin.
        target_heads (int): The desired number of heads.

    Returns:
        float: The probability of getting exactly `target_heads` heads.
    """
    # Initial distribution: 0 heads with probability 1
    dp = defaultdict(float)
    dp[0] = 1.0

    for p in probabilities:
        # Update the distribution in reverse to avoid overwriting values in the same step
        for h in range(len(dp), 0, -1):
            dp[h] = dp[h] * (1 - p) + dp[h - 1] * p
        dp[0] *= (1 - p)

    return dp[target_heads] if target_heads in dp else 0.0

def are_you_in_the_positive(xgs, goals):
    if goals > 0:
        if probability_of_heads(xgs, goals + 1) < probability_of_heads(xgs, goals) < probability_of_heads(xgs, goals - 1):
            return 1
        elif probability_of_heads(xgs, goals + 1) < probability_of_heads(xgs, goals) and probability_of_heads(xgs, goals - 1) < probability_of_heads(xgs, goals):
            return 0
        else:
            return -1
    elif goals == 0:
        if probability_of_heads(xgs, 1) < probability_of_heads(xgs, 0):
            return 0
        else:
            return -1
    elif goals == len(xgs):
        if probability_of_heads(xgs, goals - 1) < probability_of_heads(xgs, goals):
            return 0
    else:
        return -1

def least_signifcant(xgs):
    yes = []
    for x in range(0, len(xgs) + 1):
        if are_you_in_the_positive(xgs, x) == 0:
            yes.append(x)
    return xmean(yes)


def xmean(alist):
    if len(alist) == 0:
        return 0
    else:
        return sum(alist) / len(alist)


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


def fetch_and_parse_json(url):
    response = requests.get(url)
    response.raise_for_status(
    )  # Ensure we raise an error for bad status codes
    data = response.json()
    return data


def playerid():
    player = input("Name a player ")
    playerid = \
        fetch_and_parse_json("http://www.sofascore.com/api/v1/search/player-team-persons?q=" + player + "&page=0")[
            'results'][0]['entity']['id']
    return playerid


def get_all_shots(playerid):
    list_of_matches = []
    switch = "On"
    while switch == "On":
        for i in range(0, 100):
            page = "http://www.sofascore.com/api/v1/player/" + str(playerid) + "/events/last/" + str(i)
            if check_website(page):
                matchpage = fetch_and_parse_json(page)['events']
                for x in range(0, len(matchpage)):
                    if 'hasXg' in matchpage[x] and check_website("https://www.sofascore.com/api/v1/event/"+ str(matchpage[x]['id']) +"/shotmap/player/" + str(playerid)):
                        if 'xg' in fetch_and_parse_json("https://www.sofascore.com/api/v1/event/"+ str(matchpage[x]['id']) +"/shotmap/player/" + str(playerid))['shotmap'][0]:
                            list_of_matches.append(matchpage[x]['id'])
            else:
                switch = "Off"
                break
    shots = []
    others = []
    for i in range(0, len(list_of_matches)):
        if check_website("http://www.sofascore.com/api/v1/event/" + str(list_of_matches[i]) + "/shotmap/player/" + str(
                playerid)):
            match = fetch_and_parse_json(
                "http://www.sofascore.com/api/v1/event/" + str(list_of_matches[i]) + "/shotmap/player/" + str(
                    playerid))['shotmap']
            for x in range(0, len(match)):
                shots.append(match[x]['xg'])
                others.append((nonround(match[x]['xg'], 2), (nonround(find(match[x], 'xgot'), 2)), match[x]['playerCoordinates'], match[x]['goalMouthCoordinates'], match[x]['draw'], find(match[x], 'blockCoodinates'), match[x]['shotType']))
                if match[x]['shotType'] == "goal":
                    shots.append("goal")
                else:
                    shots.append("miss")
    return shots, others


def calculate_xg_stats(xg_list, outcomes_list):
    """
    Calculate the mean, standard deviation, and segment boundaries based on standard deviation.

    Parameters:
        xg_list (list or array): A list of xG values for a player.

    Returns:
        dict: A dictionary containing the mean, standard deviation,
              1 standard deviation below the mean, 1 standard deviation above the mean,
              and the dynamically created segments.
    """
    # Calculate the mean and standard deviation
    mean_xg = xmean(xg_list)
    std_dev_xg = float(np.std(xg_list))

    # Calculate 1 standard deviation below and above the mean
    one_std_below = mean_xg - std_dev_xg
    one_std_above = mean_xg + std_dev_xg

    # Create segments

    def range_decrease(smallest_chunk, x):
        return max(smallest_chunk - (x / 100), 0)

    def find_all_significant_chunks(xgs, outcomes, significance_level=0.2, min_range_size=0.02):
        def sort_data(xgs, outcomes):
            sorted_xg = sorted(xgs)
            sorted_outcomes = [x for _, x in sorted(zip(xgs, outcomes))]
            return list(zip(sorted_xg, sorted_outcomes))

        def find_largest_significant_chunk(select_range, select_outcomes, significance):
            smallest_chunk = select_range[-1]
            chunk_size = int(round(smallest_chunk, 2) * 100)
            for x in range(0, chunk_size):  # increases size of testing sector
                testing_sect = round(range_decrease(smallest_chunk, x), 2)
                if testing_sect < 0.02:
                    break
                count = 100 * round((float(smallest_chunk) - float(testing_sect)), 2)
                for i in range(0, int(round(count, 0))):  # moves sector along the section
                    lower = select_range[0] + i / 100
                    upper = round(select_range[0] + testing_sect + i / 100, 2)
                    new_range = []
                    new_outcomes = []
                    for z in range(0, len(select_range) - 1):  # creates selection with new sector values
                        if lower <= round(select_range[z], 2) <= upper:
                            new_range.append(select_range[z])
                            new_outcomes.append(select_outcomes[z])
                    goal_count = new_outcomes.count('goal')
                    if probability_of_heads(new_range, goal_count) <= significance:
                        return round(lower, 2), round(upper, 2)

        significant_chunks = []
        data = sort_data(xgs, outcomes)

        def filter_data_in_range(data, start, end):
            return [shot for shot in data if start <= shot[0] <= end]

        def recursive_find(start, end):
            # Stop if the range is smaller than the minimum allowed range size
            if end - start <= min_range_size:
                return

            # Filter data within the current range
            subset = filter_data_in_range(data, start, end)
            if not subset:
                return

            # Extract xG values and outcomes for the subset
            subset_xgs = [xg for xg, _ in subset]
            subset_outcomes = [outcome for _, outcome in subset]

            # Find the largest significant chunk within the subset
            significant_chunk = find_largest_significant_chunk(subset_xgs, subset_outcomes, significance_level)

            # If no significant chunk is found, return
            if not significant_chunk:
                return

            # Record the significant chunk
            chunk_start, chunk_end = significant_chunk
            significant_chunks.append((chunk_start, chunk_end))

            # Recursively process ranges before and after the current significant chunk
            if chunk_start > start + min_range_size:
                recursive_find(start, chunk_start - min_range_size / 2)  # Slightly overlap to avoid gaps
            if chunk_end < end - min_range_size:
                recursive_find(chunk_end + min_range_size / 2, end)  # Slightly overlap to avoid gaps

        recursive_find(0.00, 0.99)
        return significant_chunks

    sorted_pairs = sorted(zip(xg_list, outcomes_list))

    sort_xg = []
    sort_outcomes = []
    for x in sorted_pairs:
        sort_xg.append(x[0])
        sort_outcomes.append(x[1])

    data = []
    rectangle_lines = []
    # Example usage:
    for x in range(1, 9):
        data.append(find_all_significant_chunks(sort_xg, sort_outcomes,
                                                significance_level=round(float("0.000" + str(x)), 4), min_range_size=.02
                                                ))
        rectangle_lines.append("0.000" + str(x))
    for x in range(1, 9):
        data.append(find_all_significant_chunks(sort_xg, sort_outcomes,
                                                significance_level=round(float("0.00" + str(x)), 4), min_range_size=.02
                                                ))
        rectangle_lines.append("0.00" + str(x))
        data.append(find_all_significant_chunks(sort_xg, sort_outcomes,
                                                significance_level=round(float("0.00" + str(x) + "5"), 4),
                                                min_range_size=.02
                                                ))
        rectangle_lines.append("0.00" + str(x) + "5")
    for x in range(1, 9):
        data.append(find_all_significant_chunks(sort_xg, sort_outcomes,
                                                significance_level=round(float("0.0" + str(x)), 4), min_range_size=.02
                                                ))
        rectangle_lines.append("0.0" + str(x))
        data.append(find_all_significant_chunks(sort_xg, sort_outcomes,
                                                significance_level=round(float("0.0" + str(x) + "5"), 4),
                                                min_range_size=.02
                                                ))
        rectangle_lines.append("0.0" + str(x) + "5")

    rectangle_data = []
    second_rectangle_data = []
    for x in range(0, len(rectangle_lines)):
        if data[x] != data[x - 1] and 0 < len(data[x]) < 2:
            rectangle_data.append((data[x], rectangle_lines[x]))
        elif data[x] != data[x - 1] and len(data[x]) > 0:
            rectangle_data.append(([data[x][0]], rectangle_lines[x]))
            second_rectangle_data.append(([data[x][1]], rectangle_lines[x]))

    def trim(xdata, x):
        if x + 1 != len(xdata):
            lower = xdata[x][0][0][0]
            upper = xdata[x][0][0][1]
            lower2 = xdata[x + 1][0][0][0]
            upper2 = xdata[x + 1][0][0][1]
            height = [round(float(xdata[x][1]), 3), round(float(xdata[x + 1][1]), 3)]
        else:
            lower = xdata[x][0][0][0]
            lower2 = xdata[x][0][0][0]
            upper = xdata[x][0][0][1]
            upper2 = xdata[x][0][0][1]
            height = [round(float(xdata[x][1]), 3), 0.095]
        if lower < lower2 < upper:
            lower = lower2
        if upper > upper2 > lower:
            upper = upper2
        return lower, upper

    def trim2(xdata, xdata2, x):
        lower = xdata2[x][0][0][0]
        upper = xdata2[x][0][0][1]
        product = []
        for y in xdata:
            lower2 = y[0][0][0]
            upper2 = y[0][0][1]
            if lower < lower2 < upper:
                lower = lower2
            if upper > upper2 > lower:
                upper = upper2
            product.append((lower, upper))
        return lower, upper

    rectangles_list = []

    for x in range(0, len(rectangle_data)):
        rectangles_list.append(trim(rectangle_data, x))
    if len(second_rectangle_data) > 0:
        for x in range(0, len(second_rectangle_data)):
            rectangles_list.append(trim2(rectangle_data, second_rectangle_data, x))
    rectangles_list.sort(key=lambda x: x[1] - x[0])

    def remove_duplicates(original_list):
        new_list = []
        seen = set()  # To track unique (first, second) pairs

        for item in original_list:
            # Extract the first two numbers as a tuple
            pair = (item[0], item[1])

            # If the pair hasn't been added to new_list, add it
            if pair not in seen:
                seen.add(pair)
                # Append the tuple to new_list
                new_list.append(item)

        return new_list

    empty = []
    lis = []
    # Process the list
    filtered_list = remove_duplicates(rectangles_list)
    for g in filtered_list:
        if g[0] not in lis:
            lis.append(g[0])
        if g[1] not in lis:
            lis.append(g[1])

    lis.sort()
    for h in range(0, len(lis)-1):
        if h < len(lis)-2:
            empty.append((lis[h], round(lis[h + 1]-0.01, 2)))
        elif h == len(lis)-2:
            empty.append((lis[h], lis[h + 1]))
    for g in range(0, len(empty)):
        if empty[g][0] == 0.01:
            empty[g] = (0.00, empty[g][1])
        elif empty[g][1] == 0.99:
            empty[g] = (empty[g][0], 1.00)


    segments = []
    for f in empty:
        if len(f) > 1:
            segments.append((str(f[0]) + "-" + str(f[-1])))
        else:
            segments.append(str(f[0]) + "-" + str(f[0]))

    def grouped_xg_outcomes(xg_list, outcomes, segments):
        """
        Group xG values and outcomes based on the specified segments.

        Parameters:
            xg_list (list): A list of xG values.
            outcomes (list): A list of outcomes ("goal" or "miss") corresponding to the xG values.
            segments (list): A list of strings describing the ranges (e.g., '0.03-0.1').

        Returns:
            list: A list of tuples where each tuple contains the group range, the list of xG values in that range,
                  and the list of corresponding outcomes.
        """
        # Parse the group ranges into numerical boundaries
        group_boundaries = []
        for group in segments:
            lower, upper = map(float, group.split('-'))
            group_boundaries.append((lower, upper))
        # Initialize a list to store the xG values and outcomes in each group
        grouped_data = []

        # Assign each xG value and its outcome to the appropriate group
        for lower, upper in group_boundaries:
            group_xg = []
            group_outcomes = []
            for xg, outcome in zip(xg_list, outcomes):
                if lower <= round(xg, 2) <= upper or (round(xg, 2) == upper and upper == group_boundaries[-1][1]):
                    group_xg.append(xg)
                    group_outcomes.append(outcome)
            grouped_data.append((group_xg, group_outcomes))
        return grouped_data

    final_groups = []
    for group in grouped_xg_outcomes(xg_list, outcomes_list, segments):
        sub_group = ()
        for x in range(0, len(group[0])):
            sub_group += (group[0][x], group[1][x])
        final_groups.append(sub_group)

    def merge_performance_groups(grouped_xg_outcomes, segments):
        samp = []
        segments_start_end = []
        for segment in segments:
            start, end = map(float, segment.split('-'))
            segments_start_end.append((start, end))
        for b in grouped_xg_outcomes:
            bxgs = b[0::2]
            boutcomes = b[1::2]
            positive = are_you_in_the_positive(bxgs, boutcomes.count('goal'))
            if positive == 1:
                samp.append("+")
            elif positive == 0:
                samp.append("0")
            else:
                samp.append("-")
        merged_segments = []
        current_segment = segments[0]
        current_indicator = samp[0]

        for i in range(1, len(segments)):
            if samp[i] == current_indicator:
                # Merge the segments
                start = current_segment.split('-')[0]
                end = segments[i].split('-')[1]
                current_segment = f"{start}-{end}"
            else:
                # Save the current merged segment and start a new one
                merged_segments.append(current_segment)
                current_segment = segments[i]
                current_indicator = samp[i]

        # Append the last segment
        merged_segments.append(current_segment)
        return merged_segments

    final_segments = merge_performance_groups(final_groups, segments)

    # Return the results
    return {
        'mean': mean_xg,
        'standard_deviation': std_dev_xg,
        'one_std_below': one_std_below,
        'one_std_above': one_std_above,
        'segments': final_segments
    }


def split_into_groups_with_outcomes(xg_list, outcomes, groups):
    """
    Sort the xG list and corresponding outcomes, then split them into the specified groups.

    Parameters:
        xg_list (list): A list of xG values.
        outcomes (list): A list of outcomes ("goal" or "miss") corresponding to the xG values.
        groups (list): A list of strings describing the ranges (e.g., '0.03-0.1').

    Returns:
        list: A list of tuples where each tuple contains the group range, the list of xG values in that range,
              and the list of corresponding outcomes.
    """
    # Sort the xG list along with outcomes
    sorted_pairs = sorted(zip(xg_list, outcomes))

    # Parse the group ranges into numerical boundaries
    group_boundaries = []
    for group in groups:
        lower, upper = map(float, group.split('-'))
        group_boundaries.append((lower, upper))

    # Initialize a list to store the xG values and outcomes in each group
    grouped_data = []

    # Assign each xG value and its outcome to the appropriate group
    for lower, upper in group_boundaries:
        group_xg = []
        group_outcomes = []
        for xg, outcome in sorted_pairs:
            if lower <= round(xg, 2) <= upper or (round(xg, 2) == upper and upper == group_boundaries[-1][1]):
                group_xg.append(xg)
                group_outcomes.append(outcome)
        if len(group_xg) > 0:
            grouped_data.append((f"{lower}-{upper}", group_xg, group_outcomes))
    return grouped_data


def chosen_segment(chosen_segment):
    return {
        'x': chosen_segment[2].count('goal'),
        'p': chosen_segment[1],
    }

def chosen_seg_least(chosen_segment):
    return {
        'x': least_signifcant(chosen_segment[1]),
        'p': chosen_segment[1],
    }


def significance_test(chosen_segment):
    return probability_of_heads(chosen_segment['p'], chosen_segment['x'])




everything = get_all_shots(playerid())
xgs_and_outcomes = everything[0]
others = everything[1]

xgs = []

outcomes = []

for x in xgs_and_outcomes:
    if isinstance(x, float):
        xgs.append(x)
    else:
        outcomes.append(x)

sorted_pairs = sorted(zip(xgs, outcomes))

sort_xg = []
sort_outcomes = []
for x in sorted_pairs:
    sort_xg.append(x[0])
    sort_outcomes.append(x[1])

groups = calculate_xg_stats(xgs, outcomes)["segments"]


result = split_into_groups_with_outcomes(xgs, outcomes, groups)

for m in range(0, len(result)):
    print(f"Segment: {result[m][0]}")
    print(f"Number of shots: {len(result[m][1])}")
    print(f"Expected goals: {least_signifcant(result[m][1])}")
    print(f"Actual goals: {result[m][2].count('goal')}")
    print("\n")

piss = []
for j in range(0, len(result)):
    piss.append((result[j][0], len(result[j][1]), result[j][2].count('goal'), xmean(result[j][1]) * len(result[j][1]),
         significance_test(chosen_segment(result[j])), significance_test(chosen_seg_least(result[j]))))


"""
    Parameters:
        data (list): A list containing lists with data for each segment.
                     Format for each segment:
                     [xg_range, num_shots, goals_scored, xg_scored, significance]
"""
    # Names for each segment (xg_range)
segment_names = [item[0] for item in piss]
num_shots = [item[1] for item in piss]
goals_scored = [item[2] for item in piss]
xg_scored = [item[3] for item in piss]
significance = [item[4] for item in piss]
least = [item[5] for item in piss]

# Calculate bar heights and colors
heights = []
for s, g, x, l in zip(significance, goals_scored, xg_scored, least):
    if g > x:
        heights.append(1 - (s/l))
    elif g == x:
        heights.append(0)
    else:
        heights.append(-1 + (s/l))
colors = ['green' if g >= x else 'red' for g, x in zip(goals_scored, xg_scored)]

# Create the plot
fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(segment_names, heights, color=colors)

# Add shot numbers below the xg ranges
for i, shots in enumerate(num_shots):
    ax.text(i, -1.05, f'({shots})', ha='center', va='top')

# Set y-axis limits
ax.set_ylim(-1.2, 1)  # Extend the lower limit for better visibility of shot counts

# Add labels and title
ax.set_ylabel('Performance (1 - Significance)')
ax.set_title('Expected Goals Analysis')

# Show grid for clarity
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Adjust the plot layout to prevent clipping
plt.tight_layout()

plt.show()


positives = []
equals = []
negatives = []

for g in range(0, len(heights)):
    if heights[g] > 0:
        positives.append(segment_names[g])
    elif heights[g] == 0:
        equals.append(segment_names[g])
    else:
        negatives.append(segment_names[g])
shots = []
sig_info = []
answer = input("Which shots would you like to see?\na) Overperforming\nb) Normally Performing\nc) Underperforming\n\n")
if answer == "a":
    for x in positives:
        lower, upper = map(float, x.split('-'))
        for y in others:
            if lower <= y[0] <= upper:
                shots.append(y)
        for z in segment_names:
            if z == x:
                sig_info.append(piss[segment_names.index(z)])
elif answer == "b":
    for x in equals:
        lower, upper = map(float, x.split('-'))
        for y in others:
            if lower <= y[0] <= upper:
                shots.append(y)
        for z in segment_names:
            if z == x:
                sig_info.append(piss[segment_names.index(z)])
else:
    for x in negatives:
        lower, upper = map(float, x.split('-'))
        for y in others:
            if lower <= y[0] <= upper:
                shots.append(y)
        for z in segment_names:
            if z == x:
                sig_info.append(piss[segment_names.index(z)])


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
def plot_shots(shots, sig_info):
    fig, ax = draw_pitch()
    goalcount = 0
    misscount = 0

    xglis = [item[0] for item in shots]
    goals = [item[6] for item in shots].count('goal')
    for shot in shots:
        xg, xgot, player_coords, shot_coords, draw_info, block_coords, shot_type = shot

        # Filter based on xG range
        if xg != None:
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
    if goalcount >= least_signifcant(xglis):
        posneg = 1
    else:
        posneg = -1
    plt.xlim(-2, 102)  # Matches the range of the half-pitch (length)
    plt.ylim(-2, 77)  # Matches the range of the half-pitch (width)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.text(50, -7.5, "Goals scored = " + str(goalcount) + "  Score rate = " + str(round(goalcount/(goalcount+misscount), 4)) + "%" + "  Expected goals = " + str(least_signifcant(xglis)) + "  Over/Underperformance index = " + str(round(posneg * (1 - zerodivide(probability_of_heads(xglis, goals), probability_of_heads(xglis, least_signifcant(xglis)))), 7)), ha='center', fontsize=12, color='black')
    plt.show()

plot_shots(shots, sig_info)