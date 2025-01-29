import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig

def plot_confusion_matrix_with_text2(
    peg_labels,
    hole_labels,
    matrix,
    probs,
    title_text,
    output_file="confusion_matrix_with_text_Rank.png",
    tokenizer=None,
):
    # Reverse the peg_labels
    peg_labels_reversed = peg_labels[::-1]

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 12))

    # Set axis labels
    ax.set_xticks(np.arange(len(hole_labels)))
    ax.set_yticks(np.arange(len(peg_labels_reversed)))  # Use reversed peg_labels
    ax.set_xticklabels(hole_labels, rotation=90, fontsize=10)
    ax.set_yticklabels(peg_labels_reversed, fontsize=10)  # Use reversed peg_labels

    # Set x-axis labels on top
    ax.xaxis.set_tick_params(labeltop=True)  # Move x-axis labels to top
    ax.tick_params(
        axis="x", which="both", bottom=False, top=True
    )  # Ensure ticks are on top only
    ax.tick_params(axis="y", which="both", left=True, right=False)  # Ticks on the left

    # Turn grid on
    ax.set_xticks(np.arange(-0.5, len(hole_labels), 1), minor=True)
    ax.set_yticks(
        np.arange(-0.5, len(peg_labels_reversed), 1), minor=True
    )  # Use reversed peg_labels
    ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Add annotations (response + probability) and set background color
    for i in range(len(peg_labels_reversed)):
        # Get the responses and probabilities for this row
        row_responses = [
            tokenizer.decode(matrix[len(peg_labels_reversed) * i + j])
            for j in range(len(hole_labels))
        ]
        row_probs = probs[
            len(peg_labels_reversed) * i : len(peg_labels_reversed) * (i + 1)
        ]  # Probabilities for the current row

        # Debugging output
        print(f"Row {i}:")
        print(f"  hole_labels: {len(hole_labels)}")
        print(f"  row_responses: {len(row_responses)}")
        print(f"  row_probs: {len(row_probs)}")

        # Sort the cells based on probabilities for 'Yes' and 'No'
        yes_cells = [
            (j, row_probs[j], row_responses[j])
            for j in range(len(hole_labels))
            if row_responses[j] == "Yes"
        ]
        no_cells = [
            (j, row_probs[j], row_responses[j])
            for j in range(len(hole_labels))
            if row_responses[j] == "No"
        ]

        # Sort 'Yes' cells by probability in descending order
        yes_cells_sorted = sorted(yes_cells, key=lambda x: x[1], reverse=True)

        # Sort 'No' cells by probability in ascending order
        no_cells_sorted = sorted(no_cells, key=lambda x: x[1])

        # Top 3 'Yes' cells or if not enough, fill with lowest 'No' cells
        red_cells = (
            yes_cells_sorted[:3]
            if len(yes_cells_sorted) >= 3
            else yes_cells_sorted + no_cells_sorted[: 3 - len(yes_cells_sorted)]
        )

        # Add the background color based on the red cells
        for j in range(len(hole_labels)):
            response = row_responses[j]
            probability = row_probs[j]
            text = f"{response}\n{probability:.3f}"

            # Check if `red_cells` has enough elements to access safely
            is_top1 = len(red_cells) > 0 and j == red_cells[0][0]  # Top 1 cell
            is_top2_or_3 = len(red_cells) > 1 and j in [
                cell[0] for cell in red_cells[1:]
            ]  # Top 2 or 3 cells

            # Set the background color
            if is_top1:
                color = "red"  # Top 1 cell is red
            elif is_top2_or_3:
                color = "lightsalmon"  # Top 2 and 3 cells are blue
            else:
                color = "white"  # Other cells are white

            ax.add_patch(
                plt.Rectangle(
                    (j - 0.5, (len(peg_labels_reversed) - 1 - i) - 0.5),
                    1,
                    1,
                    color=color,
                )
            )

            # Add the text annotation
            ax.text(
                j,
                (len(peg_labels_reversed) - 1 - i),
                text,
                ha="center",
                va="center",
                color="black",
                fontsize=13,
            )

    # Add title and save the plot
    plt.title(title_text, fontsize=16)
    plt.savefig(output_file, bbox_inches="tight")
    plt.close()
