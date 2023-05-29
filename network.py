import networkx as nx
import bisect
from math import sqrt
import random

n_nodes = 100
rates = [50, 25, 10, 1, 0.2] # Wifi, 4G+, 4G, LTE-M, NB-IoT # Mbps

class Task(object):
    def __init__(self, task_id, dst, fn_id, msg_sz, cur_recv_rate, max_recv_rate) -> None:
        self.task_id = task_id
        self.dst = dst
        self.fn_id = fn_id
        self.msg_sz = msg_sz
        self.cur_recv_rate = cur_recv_rate
        self.max_recv_rate = max_recv_rate

    def __lt__(self, other):
        return self.max_recv_rate > other.max_recv_rate

class Event(object):
    def __init__(self, src, dst, fn_id, timestamp) -> None:
        self.src = src
        self.dst = dst
        self.fn_id = fn_id
        self.timestamp = timestamp

    def __lt__(self, other):
        return self.timestamp > other.timestamp

class EventManager(object):
    def __init__(self) -> None:
        self.event_pool = []

    def insert_event(self, e):
        bisect.insort_left(self.event_pool, e)

    def get_event(self):
        return self.event_pool.pop()

    def is_empty(self):
        return len(self.event_pool) == 0

    def del_event(self, e):
        self.event_pool.remove(e)


class Network(object):
    def __init__(self):
        p = 0.25
        sd = 1
        self.G = nx.erdos_renyi_graph(n=n_nodes, p=p, seed=sd)
        while not nx.is_connected(self.G):
            sd += 1
            self.G = nx.erdos_renyi_graph(n=n_nodes, p=p, seed=sd)

        for u, v in self.G.edges():
            self.G[u][v]['weight'] = 1.0/min(rates[int(u/20)], rates[int(v/20)])

    def gossip(self, src, tasks):
        # tasks should be sorted by the recv bandwidth
        complete_time = {}

        q = []
        t = 0
        i = 0

        while True:
            occupied_bandwidth = 0
            for current_work in q:
                occupied_bandwidth += current_work.cur_recv_rate
            available_bandwidth = rates[int(src/20)] - occupied_bandwidth

            for current_work in q:
                if current_work.cur_recv_rate < current_work.max_recv_rate:
                    if available_bandwidth >= current_work.max_recv_rate-current_work.cur_recv_rate:
                        increase = current_work.max_recv_rate-current_work.cur_recv_rate
                    else:
                        increase = available_bandwidth
                    available_bandwidth -= increase
                    current_work.cur_recv_rate += increase

            while available_bandwidth > 0 and i < len(tasks):
                tasks[i].cur_recv_rate = min(available_bandwidth, tasks[i].max_recv_rate)
                q.append(tasks[i])
                available_bandwidth -= tasks[i].cur_recv_rate
                i += 1

            wait_for_pop = []
            t0 = 1000
            for current_work in q:
                t0 = min(t0, current_work.msg_sz/current_work.cur_recv_rate)

            for index, current_work in enumerate(q):
                if t0 == current_work.msg_sz/current_work.cur_recv_rate:
                    complete_time[current_work.task_id] = t+t0
                    wait_for_pop.append(index)
                else:
                    current_work.msg_sz -= t0 * current_work.cur_recv_rate
            
            wait_for_pop.sort(reverse=True)
            for j in wait_for_pop:
                q.pop(j)
            t += t0

            if len(q) == 0 and i >= len(tasks):
                break
        
        return complete_time

    def sim_gossip_latency(self, src, msg_sz):
        clock = 0
        recv_timestamp = {src:0}
        event_manager = EventManager()
        visited = {}

        num = len(self.G.adj[src])
        candidates = random.sample(list(self.G.adj[src].keys()), int(sqrt(num)))

        visited[src] = set(candidates)
        num = len(visited[src])
        send_rate = rates[int(src/20)]
        tasks = []
        for i in range(num):
            task = Task(i, candidates[i], 0, msg_sz, 0, rates[int(candidates[i]/20)])
            bisect.insort_left(tasks, task)
        for i, t in enumerate(tasks):
            t.task_id = i
        
        complete_time = self.gossip(src, tasks)
        for tid in complete_time:
            event_manager.insert_event(Event(src, tasks[tid].dst, tasks[tid].fn_id, complete_time[tid]))

        while not event_manager.is_empty():
            ev = event_manager.get_event()
            if ev.dst not in recv_timestamp:
                recv_timestamp[ev.dst] = ev.timestamp
                if len(recv_timestamp) == n_nodes:
                    break

            if ev.dst not in visited:
                visited[ev.dst] = set()
            visited[ev.dst].add(ev.src)
            
            num_neighbor = len(self.G.adj[ev.dst])
            total = set(self.G.adj[ev.dst].keys())-visited[ev.dst]
            num = len(total)
            if sqrt(num_neighbor) >= num:
                candidates = total
            else:
                candidates = set(random.sample(total, int(sqrt(num_neighbor))))
            
            visited[ev.dst] = visited[ev.dst]|candidates

            tasks = []
            
            for cd in candidates:
                task = Task(i, cd, ev.fn_id, msg_sz, 0, rates[int(cd/20)])
                bisect.insort_left(tasks, task)
            for i, t in enumerate(tasks):
                t.task_id = i
            complete_time = self.gossip(ev.dst, tasks)
            for tid in complete_time:
                event_manager.insert_event(Event(ev.dst, tasks[tid].dst, tasks[tid].fn_id, ev.timestamp+complete_time[tid]))
            
        assert len(recv_timestamp) == n_nodes
        return recv_timestamp
    
    def sim_get_or_save_chunk(self, src, n_chunks, dsts, msg_sz):
        paths = {}
        for i in range(n_chunks):
            paths[i] = nx.dijkstra_path(self.G, src, dsts[i], weight='weight')

        clock = 0
        recv_timestamp = {}
        event_manager = EventManager()

        tasks = []
        task_dict = {}
        for i in range(n_chunks):
            if len(paths[i]) == 1:
                recv_timestamp[i] = 0
                continue
            task = Task(i, paths[i][1], i, msg_sz, 0, min(rates[int(src/20)], rates[int(paths[i][1]/20)]))
            bisect.insort_left(tasks, task)
            task_dict[i] = task
        
        complete_time = self.gossip(src, tasks)
        
        for tid in complete_time:
            event_manager.insert_event(Event(src, task_dict[tid].dst, task_dict[tid].fn_id, complete_time[tid]))

        while not event_manager.is_empty():
            ev = event_manager.get_event()
            if paths[ev.fn_id][len(paths[ev.fn_id])-1] == ev.dst:
                recv_timestamp[ev.fn_id] = ev.timestamp
            else:
                for i, v in enumerate(paths[ev.fn_id]):
                    if v == ev.dst:
                        event_manager.insert_event(Event(v, paths[ev.fn_id][i+1], ev.fn_id, ev.timestamp+msg_sz/min(rates[int(v/20)],rates[int(paths[ev.fn_id][i+1]/20)])))
                        break
        
        return recv_timestamp

# net = Network()
# # n_files = 33
# dsts = [_ for _ in range(100-34, 100)]
# # recv_timestamp = net.sim_multi_files_latency(81, n_files, dsts, 16726*8/1000/1000)
# # print(max(recv_timestamp.values()))
# recv_timestamp = net.sim_get_or_save_chunk(1, 33, dsts, 373741*8/1000/1000)
# a = list(recv_timestamp.values())
# a.sort()
# print(a)