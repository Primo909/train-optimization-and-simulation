def main():
    print("Hello from train-optimization-and-simulation!")


def optimize():
    best_scenario = scenario2
    for neighborhood in something:
        scenario = create_scenario()
        simulate(scenario)
        if check_if_better_than_previous_scenarios():
            best_scenario = scenario
        add_to_set()


def create_scenario(
    peak_frequency: float,
    non_peak_frequency: float,
    # pre-post-peak times in minutes
    first_pre_peak_time_in_minutes: float,
    first_post_peak_time_in_minutes: float,
    second_pre_peak_time_in_minutes: float,
    second_post_peak_time_in_minutes: float,
    # of carriages
    train1_first_class_carriages: int,
    train1_second_class_carriages: int,
    train2_first_class_carriages: int,
    train2_second_class_carriages: int,
    train3_first_class_carriages: int,
    train3_second_class_carriages: int,
):
    pass


def simulate(scenario):
    """Returns
    - mean of total profit
    - mean of mean of waiting time
    - mean of maximum of waiting time
    - variance of maximum of waiting time
    """
    pass


if __name__ == "__main__":
    main()
