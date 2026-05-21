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


def draw_passenger_class(origin, destination, time, u):
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
    i_class = 0

    
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
                t_first = exponential_rng(rate, u[0, i_arr])
                is_first = draw_passenger_class(origin, destination, t_first, u[1, i_class])
                push_event(PassengerGeneration(t_first, origin, destination, is_first))
                i_arr += 1
                i_class += 1

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
                next_time = e.time + exponential_rng(rate, u[0, i_arr])
                next_is_first = draw_passenger_class(e.origin, e.destination, next_time, u[1, i_class])
                push_event(
                    PassengerGeneration(
                        next_time, e.origin, e.destination, next_is_first
                    )
                )
                i_arr += 1
                i_class += 1

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

        if antithetic:
            num_runs_ind += 1

            u_antithetic = 1 - u
            result_antithetic = simulate_one_scenario(scenario=scenario, u=u_antithetic)

            stats_1 = np.asarray(result.get_statistics(), dtype=float)
            stats_2 = np.asarray(result_antithetic.get_statistics(), dtype=float)
            stats = (stats_1 + stats_2) / 2

            statistics.append(stats)
        else:
            statistics.append(result.get_statistics())
        # Check convergence: Standard Error of Mean (SEM) < PRECISION
        # SEM = sqrt(Var / n)
        convergence_statistic_id = 0
        convergence_statistic = np.array(statistics)[:, convergence_statistic_id]

        n_obs = len(statistics)
        sem_cost = np.sqrt(convergence_statistic.var() / n_obs)
        mse_cost = bootstrap_function(convergence_statistic, mean_stat, draws=1000)

        sem_mean_waiting_time = np.sqrt(
            np.var(np.array(statistics)[:, 1]) / n_obs
        )
        mse_mean_waiting_time = bootstrap_function(
            np.array(statistics)[:, 1], mean_stat, draws=1000
        ).item()

        sem_max_waiting_time = np.sqrt(
            np.var(np.array(statistics)[:, 2]) / n_obs
        )
        mse_max_waiting_time = bootstrap_function(
            np.array(statistics)[:, 2], mean_stat, draws=1000
        ).item()

        # Print progress every 10 runs
        if num_runs_ind % 10 == 0:
            sem = np.sqrt(convergence_statistic.var() / n_obs)
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


# Cost evaluation
def evaluate_profit(
    scenario: Scenario,
    seed: int,
    num_runs: int = 100,
    rng_size: int = 250000,
    antithetic: bool = True,
    verbose: bool = False,
):
    """
    Single-objective evaluation function.

    Objective:
    maximize mean net profit.
    """
    _, summary = simulate(
        seed=seed,
        scenario=scenario,
        num_runs=num_runs,
        rng_size=rng_size,
        antithetic=antithetic,
        verbose=verbose,
    )

    return summary["net_profit"]["mean"], summary

def evaluate_objective(
    objective: str,
    scenario: Scenario,
    seed: int,
    num_runs: int = 100,
    rng_size: int = 100000,
    antithetic: bool = True,
    verbose: bool = False,
):
    _, summary = simulate(
        seed=seed,
        scenario=scenario,
        num_runs=num_runs,
        rng_size=rng_size,
        antithetic=antithetic,
        verbose=verbose,
    )

    return summary[objective]["mean"], summary



from itertools import product


def neighborhood_1_peak_freq(scenario: Scenario, **kwargs):
    neighbors = []
    current_freq = scenario.train_frequency_peak

    # Generate neighbors with 5 min timesteps
    for delta in [-5, 5]:
        new_freq = max(1, current_freq + delta)
        if new_freq != current_freq:
            neighbor = Scenario(
                train_frequency_peak=new_freq,
                train_frequency_offpeak=scenario.train_frequency_offpeak,
                train_config_1=scenario.train_config_1,
                train_config_2=scenario.train_config_2,
                train_config_3=scenario.train_config_3,
                train_type_2_intervals=scenario.train_type_2_intervals,
                train_type_3_intervals=scenario.train_type_3_intervals,
                peak_times=scenario.peak_times,
            )
            neighbors.append(neighbor)

    return neighbors


def neighborhood_2_offpeak_freq(scenario: Scenario, **kwargs):

    neighbors = []
    current_freq = scenario.train_frequency_offpeak

    # Generate neighbors with 5 min timesteps
    for delta in [-5, 5]:
        new_freq = max(1, current_freq + delta)
        if new_freq != current_freq:
            neighbor = Scenario(
                train_frequency_peak=scenario.train_frequency_peak,
                train_frequency_offpeak=new_freq,
                train_config_1=scenario.train_config_1,
                train_config_2=scenario.train_config_2,
                train_config_3=scenario.train_config_3,
                train_type_2_intervals=scenario.train_type_2_intervals,
                train_type_3_intervals=scenario.train_type_3_intervals,
                peak_times=scenario.peak_times,
            )
            neighbors.append(neighbor)

    return neighbors


def neighborhood_3_train_type_1(scenario: Scenario, **kwargs):
    neighbors = []
    current_first = scenario.train_config_1.first_class_carriages
    current_second = scenario.train_config_1.second_class_carriages

    # Generate neighbors by varying first and second class carriages
    for delta_first, delta_second in [
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
        (-1, 1),
        (1, -1),
    ]:
        new_first = max(0, current_first + delta_first)
        new_second = max(0, current_second + delta_second)

        # Check carriage constraints
        if (
            new_first + new_second <= max_number_of_carriages
            and new_first + new_second > 0
            and (new_first != current_first or new_second != current_second)
        ):
            new_train_type_1 = TrainType(
                name=scenario.train_config_1.name,
                first_class_carriages=new_first,
                second_class_carriages=new_second,
            )
            neighbor = Scenario(
                train_frequency_peak=scenario.train_frequency_peak,
                train_frequency_offpeak=scenario.train_frequency_offpeak,
                train_config_1=new_train_type_1,
                train_config_2=scenario.train_config_2,
                train_config_3=scenario.train_config_3,
                train_type_2_intervals=scenario.train_type_2_intervals,
                train_type_3_intervals=scenario.train_type_3_intervals,
                peak_times=scenario.peak_times,
            )
            neighbors.append(neighbor)

    return neighbors


def neighborhood_4_train_type_2(scenario: Scenario, **kwargs):
    neighbors = []

    if scenario.train_config_2 is None:
        return neighbors

    current_first = scenario.train_config_2.first_class_carriages
    current_second = scenario.train_config_2.second_class_carriages

    # Generate neighbors by varying first and second class carriages
    for delta_first, delta_second in [
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
        (-1, 1),
        (1, -1),
    ]:
        new_first = max(0, current_first + delta_first)
        new_second = max(0, current_second + delta_second)

        # Check carriage constraints
        if (
            new_first + new_second <= max_number_of_carriages
            and new_first + new_second > 0
            and (new_first != current_first or new_second != current_second)
        ):
            new_train_type_2 = TrainType(
                name=scenario.train_config_2.name,
                first_class_carriages=new_first,
                second_class_carriages=new_second,
            )
            neighbor = Scenario(
                train_frequency_peak=scenario.train_frequency_peak,
                train_frequency_offpeak=scenario.train_frequency_offpeak,
                train_config_1=scenario.train_config_1,
                train_config_2=new_train_type_2,
                train_config_3=scenario.train_config_3,
                train_type_2_intervals=scenario.train_type_2_intervals,
                train_type_3_intervals=scenario.train_type_3_intervals,
                peak_times=scenario.peak_times,
            )
            neighbors.append(neighbor)

    return neighbors


from itertools import product


def neighborhood_5_prepost_peak_times(scenario: Scenario, **kwargs):
    neighbors = []

    current_morning_start, current_morning_end = scenario.peak_times[0]
    current_afternoon_start, current_afternoon_end = scenario.peak_times[1]

    # Deltas for:
    # morning start, morning end, afternoon start, afternoon end
    for deltas in product([-5, 0, 5], repeat=2):
        if deltas == (0, 0):
            continue

        (
            start_delta,
            end_delta,
        ) = deltas

        new_morning_start = current_morning_start + start_delta
        new_morning_end = current_morning_end + end_delta
        new_afternoon_start = current_afternoon_start + start_delta
        new_afternoon_end = current_afternoon_end + end_delta

        # Optional safety check: avoid invalid intervals
        if new_morning_start >= new_morning_end:
            continue

        if new_afternoon_start >= new_afternoon_end:
            continue

        neighbor = Scenario.create(
            peak_frequency=scenario.train_frequency_peak,
            non_peak_frequency=scenario.train_frequency_offpeak,
            train1_first_class_carriages=scenario.train_config_1.first_class_carriages,
            train1_second_class_carriages=scenario.train_config_1.second_class_carriages,
            train2_first_class_carriages=scenario.train_config_2.first_class_carriages,
            train2_second_class_carriages=scenario.train_config_2.second_class_carriages,
            first_pre_peak_time_in_minutes=PASSENGER_PEAK_TIMES[0][0]
            - new_morning_start,
            first_post_peak_time_in_minutes=new_morning_end
            - PASSENGER_PEAK_TIMES[0][1],
            second_pre_peak_time_in_minutes=PASSENGER_PEAK_TIMES[1][0]
            - new_afternoon_start,
            second_post_peak_time_in_minutes=new_afternoon_end
            - PASSENGER_PEAK_TIMES[1][1],
        )

        neighbors.append(neighbor)

    return neighbors

def variable_neighborhood_search(
    initial_scenario: Scenario,
    seed: int,
    iterations: int = 50,
    optimization_params: dict = None,
    verbose: bool = True,
):
    """
    Variable Neighborhood Search algorithm to optimize train scheduling scenario.

    Returns:
        best_scenario (Scenario): Best scenario found
        best_profit (float): Net profit of best scenario
        search_history (list): History of all scenarios evaluated
    """

    if optimization_params is None:
        optimization_params = {
            "num_runs": 100,
            "rng_size": 100000,
            "antithetic": False,
            "verbose": False,
        }

    neighborhood_structures = {
        1: neighborhood_1_peak_freq,
        2: neighborhood_2_offpeak_freq,
        3: neighborhood_3_train_type_1,
        4: neighborhood_4_train_type_2,
        5: neighborhood_5_prepost_peak_times,
    }

    current_scenario = initial_scenario
    current_profit, current_summary = evaluate_profit(
        current_scenario,
        seed=seed,
        **optimization_params,
    )

    best_scenario = current_scenario
    best_profit = current_profit

    search_history = [
        {
            "scenario": current_scenario,
            "profit": current_profit,
            "neighborhood": 0,
        }
    ]

    if verbose:
        print(f"Initial profit: {current_profit:.2f} CHF")

    neighborhood_idx = 1

    for iteration in range(iterations):
        if neighborhood_idx > len(neighborhood_structures):
            if verbose:
                print("No more neighborhoods to explore.")
            break

        neighborhood_func = neighborhood_structures[neighborhood_idx]
        neighbors = neighborhood_func(current_scenario)

        if not neighbors:
            if verbose:
                print(
                    f"Iteration {iteration + 1}: "
                    f"Neighborhood {neighborhood_idx} has no valid neighbors."
                )

            neighborhood_idx += 1
            continue

        best_neighbor = None
        best_neighbor_profit = current_profit

        for neighbor in neighbors:
            neighbor_profit, _ = evaluate_profit(
                neighbor,
                seed=seed,
                **optimization_params,
            )

            search_history.append(
                {
                    "scenario": neighbor,
                    "profit": neighbor_profit,
                    "neighborhood": neighborhood_idx,
                }
            )

            if verbose:
                print(
                    f"Scenario evaluated in neighborhood {neighborhood_idx}: "
                    f"Profit = {neighbor_profit:.2f} CHF"
                )

            if neighbor_profit > best_neighbor_profit:
                best_neighbor = neighbor
                best_neighbor_profit = neighbor_profit

        if best_neighbor is not None:
            current_scenario = best_neighbor
            current_profit = best_neighbor_profit

            if current_profit > best_profit:
                best_scenario = current_scenario
                best_profit = current_profit

            neighborhood_idx = 1

            if verbose:
                print(
                    f"Iteration {iteration + 1}: Improvement found. "
                    f"Resetting to neighborhood 1. "
                    f"Current profit: {current_profit:.2f} CHF"
                )

            continue

        if verbose:
            print(
                f"Iteration {iteration + 1}: "
                f"No improvement in neighborhood {neighborhood_idx}. "
                f"Current profit: {current_profit:.2f} CHF"
            )

        neighborhood_idx += 1

    if verbose:
        print("\nVNS complete.")
        print(f"Best profit found: {best_profit:.2f} CHF")
        print(f"Total scenarios evaluated: {len(search_history)}")

    return best_scenario, best_profit, search_history


# Multi-objective optimization


OBJECTIVE_NAMES = [
    "-mean_net_profit",
    "mean_waiting_time",
    "max_waiting_time",
]

def scenario_key(scenario: Scenario):
    """Stable key for local optimization cache."""
    return (
        scenario.train_frequency_peak,
        scenario.train_frequency_offpeak,
        scenario.train_config_1,
        scenario.train_config_2,
        scenario.train_config_3,
        scenario.train_type_2_intervals,
        scenario.train_type_3_intervals,
        scenario.peak_times,
    )


def clone_scenario(scenario: Scenario, **updates):
    data = {
        "train_frequency_peak": scenario.train_frequency_peak,
        "train_frequency_offpeak": scenario.train_frequency_offpeak,
        "train_config_1": scenario.train_config_1,
        "train_config_2": scenario.train_config_2,
        "train_config_3": scenario.train_config_3,
        "train_type_2_intervals": scenario.train_type_2_intervals,
        "train_type_3_intervals": scenario.train_type_3_intervals,
        "peak_times": scenario.peak_times,
    }

    data.update(updates)
    return Scenario(**data)

# Operational constraint: headway between consecutive trains < 1 hour.
MIN_HEADWAY = 1.0
MAX_HEADWAY = 59.0

def bounded_headway(value):
    return min(MAX_HEADWAY, max(MIN_HEADWAY, value))


def valid_train_type(first, second):
    return (
        first >= 0
        and second >= 0
        and first + second > 0
        and first + second <= max_number_of_carriages
    )


def make_train_type_neighbors(train_type: TrainType):
    neighbors = []

    for delta_first, delta_second in [
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
        (-1, 1),
        (1, -1),
    ]:
        new_first = train_type.first_class_carriages + delta_first
        new_second = train_type.second_class_carriages + delta_second

        if valid_train_type(new_first, new_second):
            if (
                new_first != train_type.first_class_carriages
                or new_second != train_type.second_class_carriages
            ):
                neighbors.append(
                    TrainType(
                        name=train_type.name,
                        first_class_carriages=new_first,
                        second_class_carriages=new_second,
                    )
                )

    return neighbors


def train_objective_vector(summary: dict):
    """
    Multi-objective cost vector to minimize.

    Objectives:
    1. Minimize -mean_net_profit
    2. Minimize mean_waiting_time
    3. Minimize max_waiting_time
    """
    cost = np.array(
        [
            -summary["net_profit"]["mean"],
            summary["mean_waiting_time"]["mean"],
            summary["max_waiting_time"]["mean"],
        ],
        dtype=float,
    )

    return np.nan_to_num(cost, nan=1e18, posinf=1e18, neginf=-1e18)


def mo_dominance(cost_x1, cost_x2, atol=1e-9):
    """
    Returns True if cost_x1 dominates cost_x2.

    Since all objectives are written as minimization objectives,
    x1 dominates x2 if:
    - x1 is no worse than x2 in all objectives;
    - x1 is strictly better than x2 in at least one objective.
    """
    cost_x1 = np.asarray(cost_x1, dtype=float)
    cost_x2 = np.asarray(cost_x2, dtype=float)

    no_worse_all = np.all(cost_x1 <= cost_x2 + atol)
    strictly_better_one = np.any(cost_x1 < cost_x2 - atol)

    return no_worse_all and strictly_better_one


def same_objective_vector(cost_x1, cost_x2, atol=1e-9):
    """Checks whether two objective vectors are numerically equivalent."""
    return np.allclose(cost_x1, cost_x2, atol=atol, rtol=0.0)


def evaluate_train_multiobjective(
    scenario: Scenario,
    seed: int,
    optimization_params: dict,
    cache: dict | None = None,
):
    """
    Evaluates one train-service scenario for multi-objective optimization.

    Returns:
    - summary: simulation summary dictionary;
    - cost_vector: multi-objective vector in minimization form.
    """
    key = (scenario_key(scenario), seed)

    if cache is not None and key in cache:
        return cache[key]

    _, summary = simulate(
        seed=seed,
        scenario=scenario,
        **optimization_params,
    )

    cost_vector = train_objective_vector(summary)

    if cache is not None:
        cache[key] = (summary, cost_vector)

    return summary, cost_vector


def generate_D_and_S_train(P: dict, cost_new: np.ndarray):
    """
    Generates the sets D and S for a new candidate solution.

    D:
        Existing Pareto solutions dominated by the new candidate.

    S:
        Existing Pareto solutions that dominate the new candidate.
        Equivalent objective vectors are also placed in S to avoid duplicates.
    """
    D = {}
    S = {}

    for key, entry in P.items():
        cost_existing = entry["cost"]

        if mo_dominance(cost_new, cost_existing):
            D[key] = entry

        if mo_dominance(cost_existing, cost_new) or same_objective_vector(
            cost_existing,
            cost_new,
        ):
            S[key] = entry

    return D, S


def update_pareto_set_train(
    P: dict,
    scenario: Scenario,
    summary: dict,
    cost_vector: np.ndarray,
    candidate_key: str,
):
    """
    Updates the Pareto set P with a new candidate scenario.

    If the candidate is dominated by at least one current Pareto solution,
    it is rejected.

    If it is not dominated, it is added to P and all solutions dominated
    by the candidate are removed.
    """
    D, S = generate_D_and_S_train(P, cost_vector)

    if S:
        return P, D, S, False

    P_updated = {key: entry for key, entry in P.items() if key not in D}

    P_updated[candidate_key] = {
        "scenario": scenario,
        "summary": summary,
        "cost": cost_vector,
    }

    return P_updated, D, S, True


def generate_train_neighbors(scenario: Scenario):
    """
    Generates all unique neighbors using the same neighborhood structures
    already used in the single-objective VNS.
    """
    neighborhood_functions = [
        neighborhood_1_peak_freq,
        neighborhood_2_offpeak_freq,
        neighborhood_3_train_type_1,
        neighborhood_4_train_type_2,
        neighborhood_5_prepost_peak_times,
    ]

    unique_neighbors = {}

    for neighborhood_func in neighborhood_functions:
        for neighbor in neighborhood_func(scenario):
            unique_neighbors[scenario_key(neighbor)] = neighbor

    return list(unique_neighbors.values())


def multiobjective_local_search_train(
    initial_scenarios: list[Scenario],
    seed: int,
    iterations: int = 100,
    optimization_params: dict | None = None,
    neighbors_per_iteration: int | None = 3,
    verbose: bool = True,
):
    """
    Multi-objective local search for the train-service optimization problem.

    Maintains a Pareto set P of non-dominated solutions.

    Objectives:
    - maximize mean net profit;
    - minimize mean waiting time;
    - minimize max waiting time.

    unserved_total is intentionally excluded from the objective vector because
    it is affected by the 24h terminal-window effect. It is kept only as a
    diagnostic output.
    """
    if optimization_params is None:
        optimization_params = {
            "num_runs": 100,
            "rng_size": 250000,
            "antithetic": True,
            "verbose": False,
        }

    rng = np.random.default_rng(seed)

    P = {}
    D_all = {}
    R_all = {}
    history = []
    evaluation_cache = {}

    # Initialization.
    for idx, scenario in enumerate(initial_scenarios):
        summary, cost_vector = evaluate_train_multiobjective(
            scenario=scenario,
            seed=seed,
            optimization_params=optimization_params,
            cache=evaluation_cache,
        )

        candidate_key = f"init_{idx}"

        P, D, S, accepted = update_pareto_set_train(
            P=P,
            scenario=scenario,
            summary=summary,
            cost_vector=cost_vector,
            candidate_key=candidate_key,
        )

        if accepted:
            D_all.update(D)
        else:
            R_all[candidate_key] = {
                "scenario": scenario,
                "summary": summary,
                "cost": cost_vector,
            }

        history.append(
            {
                "key": candidate_key,
                "scenario": scenario,
                "summary": summary,
                "cost": cost_vector,
                "accepted": accepted,
                "source": "initialization",
                "pareto_size": len(P),
            }
        )

    if verbose:
        print(f"Initial Pareto set size: {len(P)}")

    # Main local search loop.
    for iteration in range(iterations):
        if not P:
            break

        pareto_keys = list(P.keys())
        selected_key = rng.choice(pareto_keys)
        base_scenario = P[selected_key]["scenario"]

        neighbors = generate_train_neighbors(base_scenario)

        if not neighbors:
            if verbose:
                print(f"Iteration {iteration + 1}: no neighbors generated.")
            continue

        if neighbors_per_iteration is None or neighbors_per_iteration >= len(neighbors):
            candidate_neighbors = neighbors
        else:
            selected_indices = rng.choice(
                len(neighbors),
                size=neighbors_per_iteration,
                replace=False,
            )
            candidate_neighbors = [neighbors[i] for i in selected_indices]

        for local_idx, neighbor in enumerate(candidate_neighbors):
            candidate_key = f"it_{iteration + 1}_{local_idx}"

            summary, cost_vector = evaluate_train_multiobjective(
                scenario=neighbor,
                seed=seed,
                optimization_params=optimization_params,
                cache=evaluation_cache,
            )

            P_new, D, S, accepted = update_pareto_set_train(
                P=P,
                scenario=neighbor,
                summary=summary,
                cost_vector=cost_vector,
                candidate_key=candidate_key,
            )

            if accepted:
                D_all.update(D)
                P = P_new
            else:
                R_all[candidate_key] = {
                    "scenario": neighbor,
                    "summary": summary,
                    "cost": cost_vector,
                }

            history.append(
                {
                    "key": candidate_key,
                    "scenario": neighbor,
                    "summary": summary,
                    "cost": cost_vector,
                    "accepted": accepted,
                    "source": f"neighbor_of_{selected_key}",
                    "pareto_size": len(P),
                }
            )

        if verbose and (iteration + 1) % 10 == 0:
            print(
                f"Iteration {iteration + 1}/{iterations}: "
                f"Pareto set size = {len(P)}, "
                f"evaluated solutions = {len(history)}"
            )

    if verbose:
        print("\nMulti-objective local search complete.")
        print(f"Final Pareto set size: {len(P)}")
        print(f"Total evaluated solutions: {len(history)}")
        print(f"Dominated solutions removed from Pareto set: {len(D_all)}")
        print(f"Rejected/dominated candidate solutions: {len(R_all)}")

    return P, D_all, R_all, history


def pareto_set_to_dataframe(P: dict):
    rows = []

    for key, entry in P.items():
        scenario = entry["scenario"]
        summary = entry["summary"]
        cost = entry["cost"]

        peak_train = scenario.train_config_1
        offpeak_train = (
            scenario.train_config_2
            if scenario.train_config_2 is not None
            else scenario.train_config_1
        )

        rows.append(
            {
                "key": key,
                "peak_frequency": scenario.train_frequency_peak,
                "offpeak_frequency": scenario.train_frequency_offpeak,
                "peak_first_carriages": peak_train.first_class_carriages,
                "peak_second_carriages": peak_train.second_class_carriages,
                "offpeak_first_carriages": offpeak_train.first_class_carriages,
                "offpeak_second_carriages": offpeak_train.second_class_carriages,
                "morning_train_peak_start": scenario.peak_times[0][0],
                "morning_train_peak_end": scenario.peak_times[0][1],
                "evening_train_peak_start": scenario.peak_times[1][0],
                "evening_train_peak_end": scenario.peak_times[1][1],
                "mean_net_profit": summary["net_profit"]["mean"],
                "mean_waiting_time": summary["mean_waiting_time"]["mean"],
                "max_waiting_time": summary["max_waiting_time"]["mean"],
                "objective_vector": cost,
            }
        )

    df = pd.DataFrame(rows)

    if not df.empty:
        df = df.sort_values(
            by=[
                "mean_net_profit",
                "max_waiting_time",
                "mean_waiting_time",
            ],
            ascending=[False, True, True],
        ).reset_index(drop=True)

    return df


def rejected_set_to_dataframe(R_all: dict):
    rows = []

    for key, entry in R_all.items():
        scenario = entry["scenario"]
        summary = entry["summary"]

        rows.append(
            {
                "key": key,
                "peak_frequency": scenario.train_frequency_peak,
                "offpeak_frequency": scenario.train_frequency_offpeak,
                "mean_net_profit": summary["net_profit"]["mean"],
                "mean_waiting_time": summary["mean_waiting_time"]["mean"],
                "max_waiting_time": summary["max_waiting_time"]["mean"],
            }
        )

    return pd.DataFrame(rows)

def plot_pareto_profit_vs_max_wait(P: dict, R_all: dict):
    pareto_df = pareto_set_to_dataframe(P)
    rejected_df = rejected_set_to_dataframe(R_all)

    fig, ax = plt.subplots(figsize=(10, 6))

    if not rejected_df.empty:
        ax.scatter(
            rejected_df["max_waiting_time"],
            rejected_df["mean_net_profit"] / 1e6,
            alpha=0.35,
            marker="x",
            label="Rejected / dominated candidates",
        )

    if not pareto_df.empty:
        ax.scatter(
            pareto_df["max_waiting_time"],
            pareto_df["mean_net_profit"] / 1e6,
            s=80,
            label="Non-dominated Pareto set",
        )

        for _, row in pareto_df.iterrows():
            ax.annotate(
                row["key"],
                (
                    row["max_waiting_time"],
                    row["mean_net_profit"] / 1e6,
                ),
                fontsize=8,
                xytext=(4, 4),
                textcoords="offset points",
            )

    ax.set_xlabel("Max waiting time [min] — minimize")
    ax.set_ylabel("Mean net profit [million CHF] — maximize")
    ax.set_title("Pareto frontier: Profit vs Max waiting time")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig("pareto_profit_vs_max_wait.png", dpi=300)


def plot_pareto_profit_vs_mean_wait(P: dict, R_all: dict):
    pareto_df = pareto_set_to_dataframe(P)
    rejected_df = rejected_set_to_dataframe(R_all)

    fig, ax = plt.subplots(figsize=(10, 6))

    if not rejected_df.empty:
        ax.scatter(
            rejected_df["mean_waiting_time"],
            rejected_df["mean_net_profit"] / 1e6,
            alpha=0.35,
            marker="x",
            label="Rejected / dominated candidates",
        )

    if not pareto_df.empty:
        ax.scatter(
            pareto_df["mean_waiting_time"],
            pareto_df["mean_net_profit"] / 1e6,
            s=80,
            label="Non-dominated Pareto set",
        )

        for _, row in pareto_df.iterrows():
            ax.annotate(
                row["key"],
                (
                    row["mean_waiting_time"],
                    row["mean_net_profit"] / 1e6,
                ),
                fontsize=8,
                xytext=(4, 4),
                textcoords="offset points",
            )

    ax.set_xlabel("Mean waiting time [min] — minimize")
    ax.set_ylabel("Mean net profit [million CHF] — maximize")
    ax.set_title("Pareto frontier: Profit vs Mean waiting time")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig("pareto_profit_vs_mean_wait.png", dpi=300)

def plot_pareto_set_size(mo_history: list):
    mo_pareto_sizes = np.array(
        [entry["pareto_size"] for entry in mo_history],
        dtype=int,
    )

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(
        mo_pareto_sizes,
        linewidth=2,
    )

    ax.set_xlabel("Evaluated candidate")
    ax.set_ylabel("Pareto set size")
    ax.set_title("Evolution of Pareto set size")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("pareto_set_size_over_time.png", dpi=300)

######################

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

def simulate_normal_and_antithetic():
    num_runs = 100
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
    statistics, convergence = simulate(
        seed=1234,
        scenario=scenario_different_peak_times,
        num_runs=num_runs,
        rng_size=100000,
        antithetic=False,
    )
    statistics_antithetic, convergence_antithetic = simulate(
        seed=1234,
        scenario=scenario_different_peak_times,
        num_runs=num_runs,
        rng_size=100000,
        antithetic=True,
    )
    plt.plot(statistics[:, 0])
    plt.plot(statistics_antithetic[:, 0])
    plt.xlabel("Run number")
    plt.ylabel("Net profit (CHF)")
    plt.title("Convergence of Net profit")
    plt.legend(["Normal", "Antithetic"])
    plt.savefig("convergence_net_profit.png")

    fig, axes = plt.subplots(2, 3, figsize=(18, 5))
    ax1, ax2, ax3 = axes[0, 0], axes[0, 1], axes[0, 2]
    ax4, ax5, ax6 = axes[1, 0], axes[1, 1], axes[1, 2]
    x_lin = np.arange(1, len(statistics[:, 0]) + 1)
    x_lin_ant = np.arange(1, len(statistics[:, 0]) + 1, 2)
    ax1.plot(
        x_lin,
        [np.mean(statistics[:, 0][:i]) for i in range(1, len(statistics[:, 0]) + 1)],
        label="Net Profit",
    )
    ax1.plot(
        x_lin_ant,
        [
            np.mean(statistics_antithetic[:, 0][:i])
            for i in range(1, len(statistics_antithetic[:, 0]) + 1)
        ],
        label="Net Profit (Antithetic)",
    )
    ax2.plot(
        x_lin,
        [np.mean(statistics[:, 1][:i]) for i in range(1, len(statistics[:, 1]) + 1)],
        label="Mean Waiting Time",
    )
    ax2.plot(
        x_lin_ant,
        [
            np.mean(statistics_antithetic[:, 1][:i])
            for i in range(1, len(statistics_antithetic[:, 1]) + 1)
        ],
        label="Mean Waiting Time (Antithetic)",
    )
    ax3.plot(
        x_lin,
        [np.mean(statistics[:, 2][:i]) for i in range(1, len(statistics[:, 2]) + 1)],
        label="Max Waiting Time",
    )
    ax3.plot(
        x_lin_ant,
        [
            np.mean(statistics_antithetic[:, 2][:i])
            for i in range(1, len(statistics_antithetic[:, 2]) + 1)
        ],
        label="Max Waiting Time (Antithetic)",
    )
    ax1.set_title("Net Profit")
    ax2.set_title("Mean Waiting Time")
    ax3.set_title("Max Waiting Time")

    # now plot bootstrapped mean
    start = 0
    ax4.plot(
        x_lin[start:],
        [
            bootstrap_function(statistics[:, 0][:i], f_statistic=np.mean, draws=10000)
            for i in range(1, len(statistics[:, 0]) + 1)
        ],
        label="Bootstrapped MSE Net Profit",
    )
    ax4.plot(
        x_lin_ant[start:],
        [
            bootstrap_function(
                statistics_antithetic[:, 0][:i], f_statistic=np.mean, draws=10000
            )
            for i in range(1, len(statistics_antithetic[:, 0]) + 1)
        ],
        label="Bootstrapped MSE Net Profit (Antithetic)",
    )
    ax5.plot(
        x_lin[start:],
        [
            bootstrap_function(statistics[:, 1][:i], f_statistic=np.mean, draws=10000)
            for i in range(1, len(statistics[:, 1]) + 1)
        ],
        label="Bootstrapped MSE Mean Waiting Time",
    )
    ax5.plot(
        x_lin_ant[start:],
        [
            bootstrap_function(
                statistics_antithetic[:, 1][:i], f_statistic=np.mean, draws=10000
            )
            for i in range(1, len(statistics_antithetic[:, 1]) + 1)
        ],
        label="Bootstrapped MSE Mean Waiting Time (Antithetic)",
    )
    ax6.plot(
        x_lin[start:],
        [
            bootstrap_function(statistics[:, 2][:i], f_statistic=np.mean, draws=10000)
            for i in range(1, len(statistics[:, 2]) + 1)
        ],
        label="Bootstrapped MSE Max Waiting Time",
    )

    ax6.plot(
        x_lin_ant[start:],
        [
            bootstrap_function(
                statistics_antithetic[:, 2][:i], f_statistic=np.mean, draws=10000  
            )
            for i in range(1, len(statistics_antithetic[:, 2]) + 1)
        ],
        label="Bootstrapped MSE Max Waiting Time (Antithetic)",
    )
    for ax in [ax1, ax2, ax3]:
        ax.legend()
    for ax in [ax4, ax5, ax6]:
        ax.legend()
        ax.set_xlabel("Simulation Run")
        ax.set_yscale("log")
    plt.tight_layout()
    plt.savefig("convergence_and_bootstrap_mse.png")

def line_optimization():
    # dummy optimization loop to show how it could be used
    range_pre_peak = np.linspace(0, 120, 3)

    scenario_params = [
        {
            "peak_frequency": 15,
            "non_peak_frequency": 30,
            "train1_first_class_carriages": 1,
            "train1_second_class_carriages": 3,
            "train2_first_class_carriages": 1,
            "train2_second_class_carriages": 3,
            "first_pre_peak_time_in_minutes": pre_preak_time,
            "second_pre_peak_time_in_minutes": pre_preak_time,
        }
        for pre_preak_time in range_pre_peak
    ]
    scenarios = [Scenario.create(**params) for params in scenario_params]

    optimization_params = {
        "num_runs": 100,
        "rng_size": 100000,
        "antithetic": False,
        "verbose": False,
    }
    statistics_list = []
    results_list = []
    for idx, scenario in enumerate(scenarios):
        print(f"Step {idx + 1}/{len(scenarios)}")
        print(f"Running scenario: {scenario}")
        statistics, result = simulate(
            seed=1234 + idx,
            scenario=scenario,
            **optimization_params,
        )
        statistics_list.append(statistics)
        results_list.append(result)
        print(
            f"Net profit: {result['net_profit']['mean']:.2f} CHF, Mean waiting time: {result['mean_waiting_time']['mean']:.2f} min, Max waiting time: {result['max_waiting_time']['mean']:.2f} min"
        )
        print()
    print("Best Profit Scenario:")
    best = -np.inf
    for idx, result in enumerate(results_list):
        if result["net_profit"]["mean"] > best:
            best = result["net_profit"]["mean"]
            best_idx = idx
    print(f"Best scenario (Scenario {best_idx + 1}):")
    print(
        f"Best pre_peak_time: {scenario_params[best_idx]['first_pre_peak_time_in_minutes']} min"
    )
    print(f"Best result:")
    print(results_list[best_idx])

def grid_optimization():
    base_train = TrainType(
        name="base", first_class_carriages=1, second_class_carriages=3
    )
    peak_freq = np.arange(5, 46, 5)
    offpeak_freq = np.arange(5, 46, 5)

    results = np.zeros((len(peak_freq), len(offpeak_freq)))

    for i, x in enumerate(peak_freq):
        for j, y in enumerate(offpeak_freq):
            current_scenario = Scenario(
                train_frequency_peak=x,
                train_frequency_offpeak=y,
                train_config_1=base_train,
                train_config_2=base_train,
            )

            _, value = simulate(
                seed=6767,
                scenario=current_scenario,
                num_runs=10,
                rng_size=100000,
                antithetic=True,
                verbose=False,
            )

            results[i, j] = value["net_profit"]["mean"]

    best_idx = np.unravel_index(np.argmax(results), results.shape)
    best_x = peak_freq[best_idx[0]]
    best_y = offpeak_freq[best_idx[1]]
    best_profit = results[best_idx]
    print(
        f"Best average profit: {best_profit:.2f} for peak = {best_x:.2f}, offpeak = {best_y:.2f}"
    )

    # --- Plot ---
    plt.figure()
    plt.imshow(results, origin="lower", aspect="auto")
    plt.colorbar(label="Average profit")
    plt.xticks(
        range(len(offpeak_freq)), [f"{y:.2f}" for y in offpeak_freq], rotation=45
    )
    plt.yticks(range(len(peak_freq)), [f"{x:.2f}" for x in peak_freq])
    plt.xlabel("Off-peak frequency")
    plt.ylabel("Peak frequency")
    plt.title("Line search over (Peak, Off-peak)")
    plt.tight_layout()
    filename = "line_search_peak_offpeak.png"
    plt.savefig(filename)
    print(f"Saved grid search search plot to {filename}")

def single_objective_vns(num_runs: int = 100, num_iterations: int = 50):
    initial_scenario = Scenario.create(
    peak_frequency=15,
    non_peak_frequency=30,
    train1_first_class_carriages=1,
    train1_second_class_carriages=1,
    train2_first_class_carriages=1,
    train2_second_class_carriages=1,
    )

    vns_optimization_params = {
    "num_runs": num_runs,
    "rng_size": 250000,
    "antithetic": True,
    "verbose": False,
    }

    best_scenario, best_profit, history = variable_neighborhood_search(
    initial_scenario=initial_scenario,
    seed=67,
    iterations=num_iterations,
    optimization_params=vns_optimization_params,
    verbose=True,
    )

    print("\nBest scenario found:")
    print(best_scenario)
    print(f"With best profit: {best_profit:.2f} CHF")

    print("\nBest scenario summary:")
    # print(f"Mean profit: {best_summary['net_profit']['mean']:.2f} CHF")
    # print(f"Profit SEM: {best_summary['net_profit']['sem']:.2f} CHF")
    # print(f"Profit bootstrap MSE: {best_summary['net_profit']['bootstrap_mse']:.2f}")
    # print(f"Profit q05: {best_summary['net_profit']['q05']:.2f} CHF")
    # print(f"Profit q95: {best_summary['net_profit']['q95']:.2f} CHF")
    #print(f"Profit worst: {best_summary['net_profit']['max']:.2f} CHF")

    # print(f"Mean waiting time: {best_summary['mean_waiting_time']['mean']:.2f} min")
    # print(f"max waiting time: {best_summary['max_waiting_time']['mean']:.2f} min")
    # print(f"Max waiting time: {best_summary['max_waiting_time']['mean']:.2f} min")
    # print(f"Unserved passengers: {best_summary['unserved_total']['mean']:.2f}")

def multiobjective_optimization():

    initial_scenario = Scenario.create(
    peak_frequency=15,
    non_peak_frequency=30,
    train1_first_class_carriages=1,
    train1_second_class_carriages=3,
    train2_first_class_carriages=1,
    train2_second_class_carriages=3,
)

    mo_initial_scenarios = [
    initial_scenario,
    ]

    mo_optimization_params = {
    "num_runs": 10,
    "rng_size": 250000,
    "antithetic": True,
    "verbose": False,
    }


    P_mo, D_all_mo, R_all_mo, mo_history = multiobjective_local_search_train(
    initial_scenarios=mo_initial_scenarios,
    seed=123,
    iterations=10,
    optimization_params=mo_optimization_params,
    neighbors_per_iteration=3,
    verbose=True,
    )

    pareto_df = pareto_set_to_dataframe(P_mo)
    rejected_df = rejected_set_to_dataframe(R_all_mo)

    pareto_df

    plot_pareto_profit_vs_max_wait(P_mo, R_all_mo)
    plot_pareto_profit_vs_mean_wait(P_mo, R_all_mo)
    plot_pareto_set_size(mo_history)

    print("Final Pareto solutions:")

    for idx, row in pareto_df.iterrows():
        print("\n" + "-" * 80)
        print(f"Pareto solution {idx + 1}: {row['key']}")
        print(
            f"Frequency: peak {row['peak_frequency']} min, "
            f"off-peak {row['offpeak_frequency']} min"
        )
        print(
            f"Peak train: {row['peak_first_carriages']}F + {row['peak_second_carriages']}S"
        )
        print(
            f"Off-peak train: {row['offpeak_first_carriages']}F + "
            f"{row['offpeak_second_carriages']}S"
        )
        print(
            f"Train peak times: "
            f"morning ({row['morning_train_peak_start']}, "
            f"{row['morning_train_peak_end']}), "
            f"evening ({row['evening_train_peak_start']}, "
            f"{row['evening_train_peak_end']})"
        )
        print(f"Mean net profit: {row['mean_net_profit']:.2f} CHF")
        print(f"Mean waiting time: {row['mean_waiting_time']:.2f} min")
        print(f"Max waiting time: {row['max_waiting_time']:.2f} min")

def main():
    np.random.seed(1234)
    # examples_of_simulations()
    simulate_normal_and_antithetic()
    # grid_optimization()  # do grid search with only 10 runs to make it fast
    # single_objective_vns(num_runs=10, num_iterations=50)  # run VNS optimization
    # multiobjective_optimization()  # run multi-objective optimization

if __name__ == "__main__":
    main()
