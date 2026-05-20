# imports and setup
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
import heapq
import itertools
from typing import Callable
import time

stations = ["G", "L", "B", "Z"]
station_to_idx = {s: i for i, s in enumerate(stations)}
# Demand rates during peak periods [1000 passengers/h]
demand_peak = np.array(
    [
        [np.nan, 1.5, 2.2, 1.3],
        [np.nan, np.nan, 2.4, 1.1],
        [np.nan, np.nan, np.nan, 3.3],
        [np.nan, np.nan, np.nan, np.nan],
    ]
)

# Demand rates during off-peak periods [1000 passengers/h]
demand_offpeak = np.array(
    [
        [np.nan, 0.4, 0.3, 0.5],
        [np.nan, np.nan, 0.5, 0.3],
        [np.nan, np.nan, np.nan, 0.5],
        [np.nan, np.nan, np.nan, np.nan],
    ]
)

# Proportion of first class passengers during peak periods
first_class_ratio_peak = np.array(
    [
        [np.nan, 0.13, 0.15, 0.17],
        [np.nan, np.nan, 0.21, 0.18],
        [np.nan, np.nan, np.nan, 0.32],
        [np.nan, np.nan, np.nan, np.nan],
    ]
)

# Proportion of first class passengers during off-peak periods
first_class_ratio_offpeak = np.array(
    [
        [np.nan, 0.21, 0.18, 0.23],
        [np.nan, np.nan, 0.24, 0.23],
        [np.nan, np.nan, np.nan, 0.19],
        [np.nan, np.nan, np.nan, np.nan],
    ]
)

df_demand_peak = pd.DataFrame(demand_peak, index=stations, columns=stations)
df_demand_offpeak = pd.DataFrame(demand_offpeak, index=stations, columns=stations)
df_first_class_ratio_peak = pd.DataFrame(
    first_class_ratio_peak, index=stations, columns=stations
)
df_first_class_ratio_offpeak = pd.DataFrame(
    first_class_ratio_offpeak, index=stations, columns=stations
)


# Price constants
PRICE_FIRST = 40
PRICE_SECOND = 20
MISSED_COST_FIRST = 30
MISSED_COST_SECOND = 10
EMPTY_COST_FIRST = 20
EMPTY_COST_SECOND = 10
PASSENGER_PEAK_TIMES = (
    (420, 540),
    (960, 1080),
)  # Default peak times: 7-9 and 16-18


def exponential_rng(lam: float, u: float):
    """Generates exponential random number."""
    if lam <= 0:
        return float("inf")
    return -np.log(u) / lam


def get_travel_time(station):
    """Returns the travel time to the next station in minutes"""
    if station == "G":
        return 40
    if station == "L":
        return 70
    if station == "B":
        return 60
    return 0


def get_num_stations(origin, destination):
    return station_to_idx[destination] - station_to_idx[origin]


def is_peak(time_minutes, peak_times=((420, 540), (960, 1080))):
    """Checks if we are in peak hours (7-9 or 16-18)"""
    time_of_day = time_minutes % (24 * 60)
    # 7:00 = 420, 9:00 = 540, 16:00 = 960, 18:00 = 1080
    if peak_times is not None:
        for start, end in peak_times:
            if start <= time_of_day <= end:
                return True
        return False


max_number_of_carriages = 6


class Passenger:
    def __init__(self, arrival_time, origin, destination, is_first_class):
        self.arrival_time = arrival_time
        self.origin = origin
        self.destination = destination
        self.is_first_class = is_first_class
        self.missed_trains = 0


@dataclass(frozen=True)
class TrainType:
    """Train can have a maximum number of six carriages"""

    name: str
    first_class_carriages: int
    second_class_carriages: int

    def __repr__(self):
        return f"{self.name}"

    def __post_init__(self):
        if (
            self.first_class_carriages + self.second_class_carriages
            > max_number_of_carriages
        ):
            raise ValueError(f"Total carriages cannot exceed {max_number_of_carriages}")
        if self.first_class_carriages < 0 or self.second_class_carriages < 0:
            raise ValueError("Carriage numbers cannot be negative")


@dataclass
class Train:
    train_id: int
    # Standard composition: 1 first class (300) and 3 second class (1500)
    train_type: TrainType

    # Lists to keep track of passengers on board
    onboard_first: list = None
    onboard_second: list = None

    def __post_init__(self):
        self.onboard_first = []
        self.onboard_second = []

    @property
    def capacity_first(self):
        return self.train_type.first_class_carriages * 300

    @property
    def capacity_second(self):
        return self.train_type.second_class_carriages * 500


@dataclass
class Event:
    """Generic event."""

    time: float


class PassengerGeneration(Event):
    """A passenger arrives at the station"""

    def __init__(self, time, origin, destination, is_first_class):
        super().__init__(time)
        self.origin = origin
        self.destination = destination
        self.is_first_class = is_first_class


@dataclass
class TrainArrival(Event):
    """The train arrives at a station"""

    station: str
    train: Train


def get_arrival_rate(origin, destination, time):
    """Arrival rate in passengers per minute for a route."""
    i = station_to_idx[origin]
    j = station_to_idx[destination]

    if is_peak(time):
        demand = demand_peak
    else:
        demand = demand_offpeak

    total_demand_per_hour = demand[i, j] * 1000

    if np.isnan(total_demand_per_hour):
        return 0.0

    return total_demand_per_hour / 60.0


def draw_passenger_class(origin, destination, time):
    """
    Returns True for first class, False for second class.
    Class is assigned by drawing U ~ Uniform(0,1).
    """
    i = station_to_idx[origin]
    j = station_to_idx[destination]

    if is_peak(time):
        first_ratio = first_class_ratio_peak[i, j]
    else:
        first_ratio = first_class_ratio_offpeak[i, j]

    u = np.random.rand()
    return u < first_ratio


def generate_timetable(
    freq_peak, freq_offpeak
):  # Generates all train arrival events for the day
    train_events = []
    t = -120  # Trains 2 hours before
    train_id = 0

    while t < 24 * 60:
        train = Train(train_id)
        current_time = t

        # Make the train travel through the stations
        for station in stations:
            train_events.append(TrainArrival(current_time, station, train))
            current_time += get_travel_time(station)

        if is_peak(t):
            t += freq_peak
        else:
            t += freq_offpeak

        train_id += 1

    return train_events


def t_in_intervals(t, intervals: list[tuple[float]] | None):
    """Checks if time t is within any of the given intervals"""
    return any(start <= t <= end for start, end in intervals or [])


@dataclass(frozen=True)
class Scenario:
    train_frequency_peak: float
    train_frequency_offpeak: float
    train_config_1: TrainType
    train_config_2: TrainType | None = None
    train_config_3: TrainType | None = None  # maximum 3 types
    train_type_2_intervals: tuple[tuple[float, float], ...] | None = None
    train_type_3_intervals: tuple[tuple[float, float], ...] | None = None
    peak_times: tuple[tuple[float, float], ...] = (
        (420, 540),
        (960, 1080),
    )  # Default peak times: 7-9 and 16-18

    def get_timetable(
        self,
        sorted_by_time: bool = True,
    ) -> list[TrainArrival]:
        train_events = []
        t = -120
        train_id = 0
        while t < 24 * 60:
            train_type = self.train_config_1
            if t_in_intervals(t, self.train_type_2_intervals):
                train_type = self.train_config_2
            elif t_in_intervals(t, self.train_type_3_intervals):
                train_type = self.train_config_3

            if train_type is None:
                pass
            train = Train(
                train_id=train_id,
                train_type=train_type,
            )
            current_time = t

            # Make the train travel through the stations
            for station in stations:
                train_events.append(TrainArrival(current_time, station, train))
                current_time += get_travel_time(station)

            if is_peak(t):
                t += self.train_frequency_peak
            else:
                t += self.train_frequency_offpeak

            train_id += 1
        if sorted_by_time:
            train_events.sort(key=lambda event: event.time)
        return train_events

    def __post_init__(self):
        if self.train_type_2_intervals is not None:
            object.__setattr__(
                self,
                "train_type_2_intervals",
                tuple(tuple(x) for x in self.train_type_2_intervals),
            )
        else:
            if self.train_config_2 is not None:
                object.__setattr__(
                    self,
                    "train_type_2_intervals",
                    tuple(tuple(x) for x in self.peak_times),
                )
            else:
                object.__setattr__(self, "train_type_2_intervals", ())
        if self.train_type_3_intervals is not None:
            object.__setattr__(
                self,
                "train_type_3_intervals",
                tuple(tuple(x) for x in self.train_type_3_intervals),
            )
        object.__setattr__(
            self,
            "peak_times",
            tuple(tuple(x) for x in self.peak_times),
        )

    def __str__(self):
        return f"Peak frequency: {self.train_frequency_peak} min, Off-peak frequency: {self.train_frequency_offpeak} min"

    @classmethod
    def create(
        cls,
        peak_frequency: float,
        non_peak_frequency: float,
        # of carriages
        train1_first_class_carriages: int,
        train1_second_class_carriages: int,
        train2_first_class_carriages: int,
        train2_second_class_carriages: int,
        # train3_first_class_carriages: int,
        # train3_second_class_carriages: int,
        #
        # pre-post-peak times in minutes
        first_pre_peak_time_in_minutes: float = 0,
        first_post_peak_time_in_minutes: float = 0,
        second_pre_peak_time_in_minutes: float = 0,
        second_post_peak_time_in_minutes: float = 0,
    ):
        return cls(
            train_frequency_peak=peak_frequency,
            train_frequency_offpeak=non_peak_frequency,
            train_config_1=TrainType(
                name="train1",
                first_class_carriages=train1_first_class_carriages,
                second_class_carriages=train1_second_class_carriages,
            ),
            train_config_2=TrainType(
                name="train2",
                first_class_carriages=train2_first_class_carriages,
                second_class_carriages=train2_second_class_carriages,
            ),
            # train_config_3=TrainType(
            #     name="train3",
            #     first_class_carriages=train3_first_class_carriages,
            #     second_class_carriages=train3_second_class_carriages,
            # ),
            peak_times=[
                (
                    PASSENGER_PEAK_TIMES[0][0] - first_pre_peak_time_in_minutes,
                    PASSENGER_PEAK_TIMES[0][1] + first_post_peak_time_in_minutes,
                ),
                (
                    PASSENGER_PEAK_TIMES[1][0] - second_pre_peak_time_in_minutes,
                    PASSENGER_PEAK_TIMES[1][1] + second_post_peak_time_in_minutes,
                ),
            ],
        )


@dataclass
class SimulationResult:
    time_history: list[int]
    queue_history: list[int]
    waiting_times: list[float]
    total_revenue: float
    total_missed_cost: float
    total_empty_cost: float
    list_available_first: list[int]
    list_available_second: list[int]
    times_available_first: list[float]
    times_available_second: list[float]
    runtime_s: float

    def print_summary(self):
        print(f"Passengers served: {len(self.waiting_times):,.2f}")
        print(f"Average waiting time: {np.mean(self.waiting_times):.2f} minutes")
        print(f"Total revenue: {self.total_revenue:,.2f} CHF")
        print(f"Missed passenger costs: {self.total_missed_cost:,.2f} CHF")
        print(f"Empty seat costs: {self.total_empty_cost:,.2f} CHF")
        profit = self.total_revenue - self.total_missed_cost - self.total_empty_cost
        print(f"NET PROFIT: {profit:,.2f} CHF\n")
        print(f"Runtime: {self.runtime_s:.2f} seconds")

    def get_statistics(self):
        """Returns
        - net profit
        - mean waiting time
        - max waiting time
        """
        net_profit = self.total_revenue - self.total_missed_cost - self.total_empty_cost
        mean_waiting_time = np.mean(self.waiting_times).item()
        max_waiting_time = np.max(self.waiting_times).item()
        return net_profit, mean_waiting_time, max_waiting_time

    def __repr__(self):
        return f"Result(runtime_s={self.runtime_s:.2f} s, Net profit={self.total_revenue - self.total_missed_cost - self.total_empty_cost:,.2f} CHF)"


def simulate_one_scenario(scenario: Scenario, u: np.ndarray) -> SimulationResult:
    # Initialize queues at stations
    start = time.time()
    queues = {s: {"first": [], "second": []} for s in stations}

    # Priority queue for events: (time, tie_breaker, event)
    event_heap = []
    counter = itertools.count()
    i_arr = 0

    def push_event(event):
        heapq.heappush(event_heap, (event.time, next(counter), event))

    # Push all train arrivals
    for event in scenario.get_timetable(sorted_by_time=False):
        push_event(event)

    # Generate first passenger arrival event for each OD pair
    for i, origin in enumerate(stations):
        for j, destination in enumerate(stations):
            if i >= j:
                continue

            rate = get_arrival_rate(origin, destination, 0.0)
            if rate > 0:
                t_first = exponential_rng(rate, u[i_arr])
                is_first = draw_passenger_class(origin, destination, t_first)
                push_event(PassengerGeneration(t_first, origin, destination, is_first))
                i_arr += 1

    # State variables and statistics
    total_revenue = 0.0
    total_missed_cost = 0.0
    total_empty_cost = 0.0
    waiting_times = []

    time_history = []
    queue_history = []
    list_available_first = []
    list_available_second = []
    times_available_first = []
    times_available_second = []

    current_total_queue = 0

    while event_heap:
        _, _, e = heapq.heappop(event_heap)

        if e.time > 24 * 60:
            break

        if isinstance(e, PassengerGeneration):
            # Add passenger to queue
            p = Passenger(e.time, e.origin, e.destination, e.is_first_class)
            if p.is_first_class:
                queues[e.origin]["first"].append(p)
            else:
                queues[e.origin]["second"].append(p)
            current_total_queue += 1

            # Schedule next passenger generation for this OD pair
            rate = get_arrival_rate(e.origin, e.destination, e.time)
            if rate > 0:
                next_time = e.time + exponential_rng(rate, u[i_arr])
                next_is_first = draw_passenger_class(e.origin, e.destination, next_time)
                push_event(
                    PassengerGeneration(
                        next_time, e.origin, e.destination, next_is_first
                    )
                )
                i_arr += 1

        elif isinstance(e, TrainArrival):
            train = e.train
            station = e.station

            # Passengers get off
            train.onboard_first = [
                p for p in train.onboard_first if p.destination != station
            ]
            train.onboard_second = [
                p for p in train.onboard_second if p.destination != station
            ]

            # Passengers board
            if station != "Z":
                # First class boarding
                if train.train_type is None:
                    pass
                available_first = train.capacity_first - len(train.onboard_first)
                left_behind_first = []

                for p in queues[station]["first"]:
                    if available_first > 0:
                        train.onboard_first.append(p)
                        available_first -= 1
                        current_total_queue -= 1

                        waiting_times.append(e.time - p.arrival_time)
                        total_revenue += PRICE_FIRST * get_num_stations(
                            p.origin, p.destination
                        )
                        total_missed_cost += p.missed_trains * MISSED_COST_FIRST
                    else:
                        p.missed_trains += 1
                        left_behind_first.append(p)

                queues[station]["first"] = left_behind_first
                if e.time >= 0:
                    total_empty_cost += available_first * EMPTY_COST_FIRST
                    list_available_first.append(available_first)
                    times_available_first.append(e.time)

                # Second class boarding
                available_second = train.capacity_second - len(train.onboard_second)
                left_behind_second = []

                for p in queues[station]["second"]:
                    if available_second > 0:
                        train.onboard_second.append(p)
                        available_second -= 1
                        current_total_queue -= 1

                        waiting_times.append(e.time - p.arrival_time)
                        total_revenue += PRICE_SECOND * get_num_stations(
                            p.origin, p.destination
                        )
                        total_missed_cost += p.missed_trains * MISSED_COST_SECOND
                    else:
                        p.missed_trains += 1
                        left_behind_second.append(p)

                queues[station]["second"] = left_behind_second
                if e.time >= 0:
                    total_empty_cost += available_second * EMPTY_COST_SECOND
                    list_available_second.append(available_second)
                    times_available_second.append(e.time)

        if e.time >= 0:
            time_history.append(e.time)
            queue_history.append(current_total_queue)
        end = time.time()

    return SimulationResult(
        time_history=time_history,
        queue_history=queue_history,
        waiting_times=waiting_times,
        total_revenue=total_revenue,
        total_missed_cost=total_missed_cost,
        total_empty_cost=total_empty_cost,
        list_available_first=list_available_first,
        list_available_second=list_available_second,
        times_available_first=times_available_first,
        times_available_second=times_available_second,
        runtime_s=end - start,
    )


def bootstrap_function(data: np.ndarray, f_statistic: Callable, draws: int) -> float:
    """Calculates the bootstrap mse of a statistic of choice

    Keywords:
        data (array): data array.
        f_statistic: function handle calculating the statistic of interest.
        draws (int): number of bootstrap draws.

    Returns:
        mse (float): mean square error of the statistic of interest.
    """
    theta = f_statistic(data)
    emp_stats = np.zeros((draws,))
    se = np.zeros((draws,))
    for d in np.arange(draws):
        # this is a bootstrap sample
        data_draw = np.random.choice(data, size=data.shape[0], replace=True)
        # statistic is calculated using the bootstrap sample
        theta_emp = f_statistic(data_draw)
        emp_stats[d] = theta_emp
        # squared error is calculated between true value and estimate
        se[d] = (theta_emp - theta) ** 2
    return se.mean()


### statistic functions (mean, variance, quantiles, worst case)
def q025(x):
    return np.quantile(x, 0.025)


def q975(x):
    return np.quantile(x, 0.975)


def worst(x):
    return np.max(x)


def mean_stat(x):
    return np.mean(x)


def var_stat(x):
    return np.var(x, ddof=1)


def simulate(
    seed: int,
    scenario: Scenario,
    num_runs: int = 1000,
    rng_size: int = 100000,
    antithetic: bool = True,
    verbose: bool = True,
):
    """
    Returns:
    - mean of net profit
    - mean of mean of waiting times
    - mean of maximum waiting time
    """
    # Convergence parameters
    rng = np.random.default_rng(seed)

    # Initialize statistics tracking
    statistics = []
    num_runs_ind = 0

    # Main simulation loop with convergence checking
    print("Running queue simulations with independent sampling...")
    while num_runs_ind < num_runs:
        num_runs_ind += 1

        # Generate random numbers and run simulation
        u = rng.random((2, rng_size))
        result = simulate_one_scenario(scenario=scenario, u=u)
        statistics.append(result.get_statistics())

        if antithetic:
            num_runs_ind += 1
            u_antithetic = 1 - u
            result_antithetic = simulate_one_scenario(scenario=scenario, u=u_antithetic)
            statistics.append(result_antithetic.get_statistics())

        # Check convergence: Standard Error of Mean (SEM) < PRECISION
        # SEM = sqrt(Var / n)
        convergence_statistic_id = 0
        convergence_statistic = np.array(statistics)[:, convergence_statistic_id]

        sem_cost = np.sqrt(convergence_statistic.var() / num_runs_ind)
        mse_cost = bootstrap_function(convergence_statistic, mean_stat, draws=1000)

        sem_mean_waiting_time = np.sqrt(
            np.var(np.array(statistics)[:, 1]) / num_runs_ind
        )
        mse_mean_waiting_time = bootstrap_function(
            np.array(statistics)[:, 1], mean_stat, draws=1000
        ).item()

        sem_max_waiting_time = np.sqrt(
            np.var(np.array(statistics)[:, 2]) / num_runs_ind
        )
        mse_max_waiting_time = bootstrap_function(
            np.array(statistics)[:, 2], mean_stat, draws=1000
        ).item()

        # Print progress every 10 runs
        if num_runs_ind % 10 == 0:
            sem = np.sqrt(convergence_statistic.var() / num_runs_ind)
            if verbose:
                print(
                    f"  Run {num_runs_ind:3d}: Mean = {convergence_statistic.mean():.2f}, SEM = {sem:.4f}, error in % {sem / convergence_statistic.mean() * 100:.2f}%"
                )

    return np.array(statistics), {
        "net_profit": {
            "mean": np.mean(convergence_statistic).item(),
            "sem": sem_cost,
            "bootstrap_mse": mse_cost,
            "error_percent": (sem_cost / convergence_statistic.mean() * 100).item(),
        },
        "mean_waiting_time": {
            "mean": np.mean(np.array(statistics)[:, 1]).item(),
            "sem": sem_mean_waiting_time,
            "bootstrap_mse": mse_mean_waiting_time,
            "error_percent": (
                sem_cost / np.mean(np.array(statistics)[:, 1]) * 100
            ).item(),
        },
        "max_waiting_time": {
            "mean": np.mean(np.array(statistics)[:, 2]).item(),
            "sem": sem_max_waiting_time,
            "bootstrap_mse": mse_max_waiting_time,
            "error_percent": (
                sem_cost / np.mean(np.array(statistics)[:, 2]) * 100
            ).item(),
        },
    }


def examples_of_simulations():
    # example scenarios
    base_train = TrainType(
        name="base", first_class_carriages=1, second_class_carriages=3
    )
    short_train = TrainType(
        name="short", first_class_carriages=1, second_class_carriages=1
    )
    long_train = TrainType(
        name="long", first_class_carriages=1, second_class_carriages=5
    )

    scenario_1 = Scenario(
        train_frequency_peak=20,
        train_frequency_offpeak=20,
        train_config_1=base_train,
    )

    scenario_2 = Scenario(
        train_frequency_peak=15,
        train_frequency_offpeak=30,
        train_config_1=base_train,
    )

    scenario_train_every_minute = Scenario(
        train_frequency_peak=1,
        train_frequency_offpeak=1,
        train_config_1=base_train,
    )

    scenario_train_every_minute = Scenario(
        train_frequency_peak=1,
        train_frequency_offpeak=1,
        train_config_1=base_train,
    )

    scenario_different_peak_times = Scenario.create(
        peak_frequency=15,
        non_peak_frequency=30,
        train1_first_class_carriages=1,
        train1_second_class_carriages=3,
        train2_first_class_carriages=1,
        train2_second_class_carriages=3,
        first_pre_peak_time_in_minutes=30,
        second_pre_peak_time_in_minutes=30,
    )

    statistics = [[] for _ in range(4)]
    queue_history = [[] for _ in range(4)]
    time_history = [[] for _ in range(4)]
    for idx, scenario in enumerate(
        [
            scenario_1,
            scenario_2,
            scenario_train_every_minute,
            scenario_different_peak_times,
        ]
    ):
        print(f"Running scenario: {scenario}")
        result = simulate_one_scenario(scenario, u=np.random.rand(100000))
        statistics[idx] = result.get_statistics()
        queue_history[idx] = result.queue_history
        time_history[idx] = result.time_history

    fig, axes = plt.subplots(figsize=(18, 5), sharey=True)
    axes = [axes]

    for idx, (name, scenario_stats, qh, th, peak_times) in enumerate(
        zip(
            [
                "Scenario 1",
                "Scenario 2",
                "Scenario 3 (Train every minute)",
                "Scenario 4 (Different peak times)",
            ],
            statistics,
            queue_history,
            time_history,
            [
                scenario.peak_times
                for scenario in [
                    scenario_1,
                    scenario_2,
                    scenario_train_every_minute,
                    scenario_different_peak_times,
                ]
            ],
        )
    ):
        print(name)
        scenario_stats = np.array(scenario_stats)
        print(f"Net profit: {np.mean(scenario_stats[0]).item():.2e} CHF")
        print(f"Mean waiting time: {np.mean(scenario_stats[1]).item():.2f}")
        print(f"Max waiting time: {np.mean(scenario_stats[2]).item():.2f}")
        print()
        axes[0].plot(np.array(th) / 60, qh, label=name)
        axes[0].set_title(name)
        axes[0].set_xlabel("Event number")
        axes[0].set_ylabel("Queue length")
        axes[0].set_xlim(0, 24)
        for idx_peaks, (peak_start, peak_end) in enumerate(peak_times):
            peak_start_hours = peak_start / 60
            peak_end_hours = peak_end / 60
            axes[0].axvspan(
                peak_start_hours,
                peak_end_hours,
                color="gray",
                alpha=0.4,
                label="Peak period of Trains" if idx == 0 and idx_peaks == 0 else None,
            )
        for idx_peaks, (peak_start, peak_end) in enumerate(PASSENGER_PEAK_TIMES):
            peak_start_hours = peak_start / 60
            peak_end_hours = peak_end / 60
            axes[0].axvspan(
                peak_start_hours,
                peak_end_hours,
                color="red",
                alpha=0.2,
                label="Peak period of Passengers"
                if idx == 0 and idx_peaks == 0
                else None,
            )
        axes[0].legend()
    plt.tight_layout()
    plt.savefig("queue_length_over_time_four_example_scenarios.png")


def main():
    np.random.seed(1234)
    examples_of_simulations()


if __name__ == "__main__":
    main()
