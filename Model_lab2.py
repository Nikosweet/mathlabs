import math
import random
from collections import deque

NUM_DEVICES = 5
BUFFER_SIZE = 3
SIMULATION_DETAILS = 10000

processed = 0

lost_all = 0
total_arrived = 0


class Device:
    def __init__(self, id_num):
        self.id = id_num
        self.busy_until = 0.0
        self.failed_until = 0.0
        self.buffer = deque()
        self.total_served = 0
        self.total_failures = 0

    def is_free(self, current_time):
        if current_time < self.failed_until:
            return False
        return current_time >= self.busy_until

    def try_add_to_buffer(self, arrival_time):
        if len(self.buffer) < BUFFER_SIZE:
            self.buffer.append(arrival_time)
            return True
        else:
            return False

    def start_service_from_buffer(self, current_time):
        if len(self.buffer) > 0:
            arrival_time = self.buffer.popleft()

            service = service_duration()
            if need_reconfiguration():
                service += reconfiguration_time()

            if not is_working():
                self.total_failures += 1
                self.failed_until = current_time + fix_time()
                self.busy_until = current_time
                self.buffer.appendleft(arrival_time)
                return False

            self.busy_until = current_time + service
            self.total_served += 1
            return True
        return False

def new_detail_time():
    return random.uniform(3, 7)


def to_next():
    return 1.0


def service_duration():
    return -5 * math.log(random.random())


def need_reconfiguration():
    return random.random() <= 0.2


def reconfiguration_time():
    return -4 * math.log(random.random())


def is_working():
    return random.random() >= 0.1


def fix_time():
    return max(0.1, random.gauss(15, 3))


devices = [Device(i) for i in range(NUM_DEVICES)]

current_time = 0.0
next_arrival = new_detail_time()

while total_arrived < SIMULATION_DETAILS:
    current_time = next_arrival
    total_arrived += 1

    detail_arrival_to_line = current_time
    detail_processed = False

    for i, device in enumerate(devices):
        arrival_to_device = detail_arrival_to_line + i * to_next()

        if device.is_free(arrival_to_device):
            device.start_service_from_buffer(arrival_to_device)

        if device.is_free(arrival_to_device) and len(device.buffer) == 0:
            service = service_duration()
            if need_reconfiguration():
                service += reconfiguration_time()

            if not is_working():
                device.total_failures += 1
                device.failed_until = arrival_to_device + fix_time()
                device.busy_until = arrival_to_device
                if device.try_add_to_buffer(arrival_to_device):
                    pass
                else:
                    pass
            else:
                device.busy_until = arrival_to_device + service
                device.total_served += 1
                detail_processed = True
                break
        else:
            if device.try_add_to_buffer(arrival_to_device):
                detail_processed = True
                break
            else:
                continue

    if not detail_processed:
        lost_all += 1

    next_arrival = current_time + new_detail_time()

max_simulation_time = max(d.busy_until for d in devices)
max_simulation_time = max(max_simulation_time, max(d.failed_until for d in devices))

current_time = next_arrival
while any(len(d.buffer) > 0 or not d.is_free(current_time) for d in devices):
    next_event_time = float('inf')

    for device in devices:
        if device.failed_until > current_time:
            next_event_time = min(next_event_time, device.failed_until)
        if device.busy_until > current_time:
            next_event_time = min(next_event_time, device.busy_until)

    if next_event_time == float('inf'):
        break

    current_time = next_event_time

    for device in devices:
        if current_time >= device.failed_until and device.failed_until > 0:
            device.failed_until = 0

        if device.is_free(current_time):
            device.start_service_from_buffer(current_time)

total_processed = sum(d.total_served for d in devices)


print(NUM_DEVICES)
print(BUFFER_SIZE)
print(total_arrived)
print(total_processed, total_processed / total_arrived * 100)
print(lost_all)
