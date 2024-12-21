import matplotlib.pyplot as plt
import numpy as np
import requests
from scipy.stats import binom
from itertools import combinations

def new_sig_test(xgs, outcomes):
    goals = outcomes.count('goal')
    misses = outcomes.count('miss')
    n = len(outcomes)
    heads_positions = list(combinations(range(n), goals))
    total_sig = 0
    for positions in heads_positions:
        combo = [1 - p for p in xgs]
        for pos in positions:
            combo[pos] = xgs[pos]
        total_sig += sum(combo)
    return total_sig


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
                    if 'hasXg' in matchpage[x]:
                        list_of_matches.append(matchpage[x]['id'])
            else:
                switch = "Off"
                break
    shots = []
    for i in range(0, len(list_of_matches)):
        if check_website("http://www.sofascore.com/api/v1/event/" + str(list_of_matches[i]) + "/shotmap/player/" + str(
                playerid)):
            match = fetch_and_parse_json(
                "http://www.sofascore.com/api/v1/event/" + str(list_of_matches[i]) + "/shotmap/player/" + str(
                    playerid))['shotmap']
            for x in range(0, len(match)):
                shots.append(match[x]['xg'])
                if match[x]['shotType'] == "goal":
                    shots.append("goal")
                else:
                    shots.append("miss")
    return shots


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
    def mean(alist):
        return sum(alist) / len(alist) if alist else 0

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
                    sample_size = len(new_range)
                    probability = mean(new_range)
                    if binom.pmf(goal_count, sample_size, probability) <= significance:
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


    """one_to_100 = [0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16,
                  0.17, 0.18, 0.19, 0.20, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.30, 0.31, 0.32, 0.33,
                  0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.40, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.50,
                  0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.60, 0.61, 0.62, 0.63, 0.64, 0.65, 0.66, 0.67,
                  0.68, 0.69, 0.70, 0.71, 0.72, 0.73, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79, 0.80, 0.81, 0.82, 0.83, 0.84,
                  0.85, 0.86, 0.87, 0.88, 0.89, 0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]
    for c in filtered_list:
        tup = ()
        for d in range(int((100 * c[0])), int((100 * c[1]))):
            e = d / 100
            if e in one_to_100:
                tup += (e,)
                one_to_100.remove(e)
        if len(tup) > 1:
            empty.append(tup)
    empty.sort(key=lambda x: x[0])"""
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
            if sum(bxgs) <= boutcomes.count('goal'):
                samp.append("+")
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
            if lower <= round(xg, 2) < upper or (round(xg, 2) == upper and upper == group_boundaries[-1][1]):
                group_xg.append(xg)
                group_outcomes.append(outcome)
        grouped_data.append((f"{lower}-{upper}", group_xg, group_outcomes))
    return grouped_data


def chosen_segment(chosen_segment):
    return {
        'x': chosen_segment[2].count('goal'),
        'n': len(chosen_segment[2]),
        'p': xmean(chosen_segment[1]),
    }


def significance_test(chosen_segment):
    return binom.pmf(chosen_segment['x'], chosen_segment['n'], chosen_segment['p'])


namedplayerid = playerid()

xgs_and_outcomes = get_all_shots(namedplayerid)

xgs = []

outcomes = []

for x in xgs_and_outcomes:
    if x == len(xgs_and_outcomes) // 10:
        print("Loading...")
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
    print(f"Number of goals: {result[m][2].count('goal')}")
    print(f"Expected goals: {xmean(result[m][1]) * len(result[m][1])}")
    print(f"Actual goals: {result[m][2].count('goal')}")
    print(f"Significance: {significance_test(chosen_segment(result[m]))}")
    print(f"Significant enough? {significance_test(chosen_segment(result[m])) <= 0.1}")
    print("\n")

piss = []
for j in range(0, len(result)):
    piss.append((result[j][0], len(result[j][1]), result[j][2].count('goal'), xmean(result[j][1]) * len(result[j][1]),
         significance_test(chosen_segment(result[j]))))

"""first = [result[0][0], len(result[0][1]), result[0][2].count('goal'), xmean(result[0][1]) * len(result[0][1]),
         significance_test(chosen_segment(result[0]))]
second = [result[1][0], len(result[1][1]), result[1][2].count('goal'), xmean(result[1][1]) * len(result[1][1]),
          significance_test(chosen_segment(result[1]))]
third = [result[2][0], len(result[2][1]), result[2][2].count('goal'), xmean(result[2][1]) * len(result[2][1]),
         significance_test(chosen_segment(result[2]))]
fourth = [result[3][0], len(result[3][1]), result[3][2].count('goal'), xmean(result[3][1]) * len(result[3][1]),
          significance_test(chosen_segment(result[3]))]
fifth = [result[4][0], len(result[4][1]), result[4][2].count('goal'), xmean(result[4][1]) * len(result[4][1]),
         significance_test(chosen_segment(result[4]))]
sixth = [result[5][0], len(result[5][1]), result[5][2].count('goal'), xmean(result[5][1]) * len(result[5][1]),
         significance_test(chosen_segment(result[5]))]
seventh = [result[6][0], len(result[6][1]), result[6][2].count('goal'), xmean(result[6][1]) * len(result[6][1]),
           significance_test(chosen_segment(result[6]))]"""


def plot_xg_barchart(data):
    """
    Parameters:
        data (list): A list containing lists with data for each segment.
                     Format for each segment:
                     [xg_range, num_shots, goals_scored, xg_scored, significance]
    """
    # Names for each segment (xg_range)
    segment_names = [item[0] for item in data]
    num_shots = [item[1] for item in data]
    goals_scored = [item[2] for item in data]
    xg_scored = [item[3] for item in data]
    significance = [item[4] for item in data]

    # Calculate bar heights and colors
    heights = [1 - s if g >= x else -(1 - s) for s, g, x in zip(significance, goals_scored, xg_scored)]
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


##data = [first, second, third, fourth, fifth, sixth, seventh]

# Plot the chart
plot_xg_barchart(piss)
