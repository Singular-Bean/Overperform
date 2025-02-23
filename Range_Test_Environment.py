from matplotlib import pyplot as plt
from itertools import combinations
import math
from collections import defaultdict


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


def new_sig_test(xgs, outcomes):
    goals = outcomes.count('goal')
    n = len(outcomes)
    heads_positions = list(combinations(range(n), goals))
    total_sig = 0
    for positions in heads_positions:
        combo = [1 - p for p in xgs]
        for pos in positions:
            combo[pos] = xgs[pos]
        total_sig += math.prod(combo)
    return total_sig



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


egxgs = [0.046646554023027, 0.7884, 0.078808180987835, 0.041611388325691, 0.029515432193875, 0.035937268286943,
         0.066927783191204, 0.091790705919266, 0.038542382419109, 0.049081012606621, 0.41868832707405,
         0.039774421602488, 0.061227269470692, 0.26950958371162, 0.14739675819874, 0.54234021902084, 0.016091531142592,
         0.18338394165039, 0.04551150277257, 0.7884, 0.22148741781712, 0.60982227325439, 0.02375328168273,
         0.093795634806156, 0.30120280385017, 0.0089707020670176, 0.055181529372931, 0.24091856181622, 0.10318325459957,
         0.058561906218529, 0.088545948266983, 0.13011384010315, 0.049194060266018, 0.021351793780923, 0.11083568632603,
         0.094914741814137, 0.054173707962036, 0.052504606544971, 0.035400878638029, 0.01773333363235, 0.88510179519653,
         0.0082504590973258, 0.019121890887618, 0.13227415084839, 0.083388969302177, 0.25983014702797,
         0.040193412452936, 0.96432417631149, 0.08450660854578, 0.26077708601952, 0.031161366030574, 0.45485639572144,
         0.032316740602255, 0.7884, 0.027790050953627, 0.7884, 0.080117680132389, 0.042569793760777, 0.026276465505362,
         0.10427866876125, 0.035057384520769, 0.041881110519171, 0.036687377840281, 0.0089496457949281,
         0.047903694212437, 0.045588806271553, 0.10947313159704, 0.063959330320358, 0.037396833300591, 0.7884,
         0.024532500654459, 0.031302567571402, 0.075898438692093, 0.33602851629257, 0.024492297321558,
         0.039705295115709, 0.02463193051517, 0.017619527876377, 0.014163195155561, 0.04999190568924, 0.016739696264267,
         0.028673147782683, 0.15364716947079, 0.037845890969038, 0.045189052820206, 0.022403724491596,
         0.056075029075146, 0.3770823776722, 0.048774484544992, 0.091550670564175, 0.7884, 0.037647556513548,
         0.17063733935356, 0.14034099876881, 0.01631448790431, 0.018841784447432, 0.036065246909857, 0.7884,
         0.02015184238553, 0.034777779132128, 0.036817517131567, 0.030080771073699, 0.083395332098007,
         0.026432348415256, 0.7884, 0.020327914506197, 0.04458474740386, 0.7884, 0.40758097171783, 0.31838962435722,
         0.91693359613419, 0.066953293979168, 0.34248355031013, 0.039230190217495, 0.13622486591339, 0.011680975556374,
         0.036198306828737, 0.040803760290146, 0.040936678647995, 0.050268709659576, 0.14805527031422, 0.12426443397999,
         0.95646321773529, 0.095466710627079, 0.074367046356201, 0.020672731101513, 0.17348927259445, 0.16855727136135,
         0.14443597197533, 0.11411913484335, 0.17126278579235, 0.015475064516068, 0.077742166817188, 0.018055470660329,
         0.035631462931633, 0.048467449843884, 0.029671724885702, 0.037354286760092, 0.10863254964352, 0.24825152754784,
         0.023366576060653, 0.15896591544151, 0.036055285483599, 0.01486314740032, 0.057369936257601, 0.034806087613106,
         0.031890161335468, 0.096118666231632, 0.038311287760735, 0.37320020794868, 0.067297570407391,
         0.044712103903294, 0.7884, 0.4840247631073, 0.038994051516056, 0.024168433621526, 0.067008964717388,
         0.021628618240356, 0.10435865819454, 0.02199842967093, 0.044905763119459, 0.019110256806016, 0.087853416800499,
         0.47518208622932, 0.025860644876957, 0.060948383063078, 0.055599618703127, 0.041604541242123,
         0.018469529226422, 0.96439361572266, 0.015317994169891, 0.021092317998409, 0.037737961858511,
         0.019221840426326, 0.02160851098597, 0.056757397949696, 0.034401867538691, 0.65143877267838, 0.15712393820286,
         0.21138799190521, 0.14817102253437, 0.023838168010116, 0.11482284218073, 0.055876463651657, 0.029312014579773,
         0.073976576328278, 0.051987290382385, 0.043128877878189, 0.081994205713272]
egoutcomes = ['miss', 'goal', 'miss', 'goal', 'miss', 'miss', 'miss', 'miss', 'goal', 'miss', 'miss', 'miss', 'miss',
            'goal', 'miss', 'miss', 'miss', 'goal', 'goal', 'goal', 'goal', 'miss', 'miss', 'miss', 'miss', 'miss',
            'miss', 'miss', 'miss', 'miss', 'miss', 'miss', 'goal', 'miss', 'miss', 'miss', 'miss', 'miss', 'miss',
            'miss', 'miss', 'miss', 'goal', 'miss', 'miss', 'miss', 'miss', 'goal', 'miss', 'miss', 'miss', 'miss',
            'miss', 'goal', 'miss', 'goal', 'miss', 'miss', 'miss', 'miss', 'miss', 'miss', 'miss', 'miss', 'miss',
            'miss', 'miss', 'goal', 'miss', 'goal', 'miss', 'miss', 'miss', 'goal', 'miss', 'miss', 'miss', 'miss',
            'miss', 'miss', 'miss', 'miss', 'miss', 'miss', 'goal', 'miss', 'miss', 'goal', 'miss', 'miss', 'goal',
            'miss', 'miss', 'miss', 'miss', 'miss', 'goal', 'goal', 'miss', 'miss', 'miss', 'miss', 'miss', 'miss',
            'goal', 'miss', 'miss', 'goal', 'miss', 'goal', 'goal', 'goal', 'miss', 'miss', 'miss', 'miss', 'miss',
            'miss', 'miss', 'miss', 'miss', 'miss', 'miss', 'miss', 'miss', 'miss', 'goal', 'miss', 'miss', 'miss',
            'miss', 'miss', 'goal', 'miss', 'miss', 'miss', 'miss', 'miss', 'miss', 'miss', 'miss', 'miss', 'miss',
            'miss', 'miss', 'miss', 'miss', 'miss', 'miss', 'miss', 'miss', 'miss', 'goal', 'miss', 'miss', 'miss',
            'goal', 'miss', 'miss', 'miss', 'miss', 'miss', 'miss', 'goal', 'miss', 'miss', 'miss', 'miss', 'miss',
            'goal', 'miss', 'miss', 'goal', 'miss', 'miss', 'miss', 'miss', 'miss', 'miss', 'miss', 'miss', 'miss',
            'miss', 'miss', 'miss', 'miss', 'miss', 'miss', 'miss']

sorted_pairs = sorted(zip(egxgs, egoutcomes))

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
                                            significance_level=round(float("0.0" + str(x) + "5"), 4), min_range_size=.02
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
# Process the list
filtered_list = remove_duplicates(rectangles_list)
one_to_100 = [0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16,
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
empty.sort(key=lambda x: x[0])
segments = []
for f in empty:
    if len(f) > 1:
        segments.append((str(f[0]) + "-" + str(f[-1])))
    else:
        segments.append(str(f[0]) + "-" + str(f[0]))
print(segments)
"""

"""# Example data: significance levels and associated ranges
significance_levels = [
    0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009, 0.001, 0.0015, 0.002, 0.0025,
    0.003, 0.0035, 0.004, 0.0045, 0.005, 0.0055, 0.006, 0.0065, 0.007, 0.0075, 0.008, 0.0085, 0.009, 0.0095,
    0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06, 0.065, 0.07, 0.075, 0.08, 0.085,
    0.09, 0.095
]


# Cap all values at 1
def cap_values(ranges):
    return [(max(0, min(1, start)), max(0, min(1, end))) for start, end in ranges]


# Prepare the plot
plt.figure(figsize=(10, 8))

for i, ranges in enumerate(data[:-1]):
    next_level = significance_levels[i + 1] if i + 1 < len(significance_levels) else 1
    capped_ranges = cap_values(ranges)
    for start, end in capped_ranges:
        # Draw a rectangle for the range
        plt.fill_betweenx(
            [significance_levels[i], next_level],
            start, end,
            color='blue', alpha=0.3
        )

# Labels and styling
plt.xlabel("xG Range")
plt.ylabel("Significance Level")
plt.title("Significant Ranges Across Varying Significance Levels")
plt.xlim(0, 1)
plt.ylim(min(significance_levels), max(significance_levels))
plt.gca().invert_yaxis()  # Invert y-axis for better readability
plt.grid(alpha=0.5)
plt.show()



result = probability_of_heads(egxgs, egoutcomes.count('goal'))
print(f"Probability of getting {egoutcomes.count('goal')} heads: {result}")
