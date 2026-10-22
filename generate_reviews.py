import argparse
import csv
import random
from pathlib import Path

ROOT = Path(__file__).parent
THEMES = {
    "audio_quality": (
        ["headphones", "earbuds", "speaker"],
        ["clear", "balanced", "rich", "crisp"],
        ["music sounds detailed", "the bass is surprisingly good", "podcasts are easy to hear"],
    ),
    "battery": (
        ["headphones", "earbuds", "portable speaker"],
        ["excellent", "short", "reliable", "disappointing"],
        ["it lasts through my commute", "I have to charge it every evening", "the battery indicator is helpful"],
    ),
    "delivery": (
        ["package", "order", "parcel"],
        ["quick", "late", "careful", "confusing"],
        ["tracking updates were useful", "the box arrived in good condition", "delivery took longer than promised"],
    ),
    "comfort": (
        ["headphones", "earbuds", "headset"],
        ["comfortable", "tight", "lightweight", "stiff"],
        ["I can wear it for hours", "the ear cushions need improvement", "it fits well during a workout"],
    ),
    "connectivity": (
        ["Bluetooth connection", "wireless pairing", "phone connection"],
        ["stable", "unreliable", "quick", "frustrating"],
        ["pairing took only a few seconds", "the signal drops in another room", "switching devices is straightforward"],
    ),
    "customer_support": (
        ["support team", "service agent", "help desk"],
        ["helpful", "slow", "professional", "unresponsive"],
        ["my issue was resolved on the first call", "I had to repeat the problem twice", "the replacement process was clear"],
    ),
}
OPENERS = [
    "Overall,", "For the price,", "After a week,", "In daily use,",
    "Honestly,", "Compared with my old one,", "As a gift,", "So far,",
]
ENDINGS = [
    "I would consider buying it again.",
    "That detail mattered more than I expected.",
    "This is the part I noticed most often.",
    "It was the deciding factor for me.",
    "The experience was different from what I expected.",
]


def generate_rows(per_topic: int, seed: int):
    if per_topic < 1:
        raise ValueError("per_topic must be at least 1.")
    rng = random.Random(seed)
    rows = []
    for theme, (products, adjectives, details) in THEMES.items():
        for _ in range(per_topic):
            product = rng.choice(products)
            adjective = rng.choice(adjectives)
            detail = rng.choice(details)
            opener = rng.choice(OPENERS)
            ending = rng.choice(ENDINGS)
            rows.append(
                {
                    "review_id": f"synthetic-{len(rows) + 1:05d}",
                    "review": (
                        f"{opener} the {product} is {adjective}; "
                        f"{detail}. {ending}"
                    ),
                    "theme": theme,
                    "is_synthetic": "true",
                }
            )
    rng.shuffle(rows)
    return rows


def main():
    parser = argparse.ArgumentParser(description="Generate fictional review examples.")
    parser.add_argument("--per-topic", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "generated_reviews.csv",
    )
    args = parser.parse_args()
    rows = generate_rows(args.per_topic, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} explicitly synthetic reviews to {args.output}")


if __name__ == "__main__":
    main()
