import argparse

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--name", type=str, default="World", help="Name of the simulation world.")
    parser.add_argument("--duration", type=int, default=100, help="Duration of the simulation in seconds.")

    args = parser.parse_args()
    print(f"Simulation Name: {args.name}")
    print(f"Simulation Duration: {args.duration} seconds")

if __name__ == "__main__":
    main()