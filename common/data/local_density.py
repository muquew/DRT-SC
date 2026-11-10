from collections import Counter, defaultdict
import math


def choose_grid_district(requests, workers, *, radius_m=900., required_tasks=108, required_workers=10):

    if type(radius_m)not in(int,float)or not math.isfinite(radius_m)or radius_m<=0:
        raise ValueError('radius_m must be positive and finite')
    requests, workers = list(requests), list(workers)
    counts = Counter()
    for key,pickup,delivery in requests:
        x0,x1 = math.floor((pickup[0]-radius_m)/radius_m), math.floor((pickup[0]+radius_m)/radius_m)
        y0,y1 = math.floor((pickup[1]-radius_m)/radius_m), math.floor((pickup[1]+radius_m)/radius_m)
        for x in range(x0,x1+1):
            for y in range(y0,y1+1):
                center = (x+.5)*radius_m, (y+.5)*radius_m
                if math.dist(center,pickup) <= radius_m and math.dist(center,delivery) <= radius_m:
                    counts[x,y] += 1
    worker_cells = defaultdict(list)
    for key,xy in workers:
        for x in range(math.floor((xy[0]-radius_m)/radius_m),math.floor((xy[0]+radius_m)/radius_m)+1):
            for y in range(math.floor((xy[1]-radius_m)/radius_m),math.floor((xy[1]+radius_m)/radius_m)+1):
                center = (x+.5)*radius_m,(y+.5)*radius_m
                if math.dist(center,xy) <= radius_m:
                    worker_cells[x,y].append(key)
    ranked = []
    for cell,count in counts.most_common():
        center = tuple((x+.5)*radius_m for x in cell)
        matching_workers = worker_cells[cell]
        ranked.append((count,len(matching_workers),cell,matching_workers))
    ranked.sort(key=lambda r:(-r[0],-r[1],r[2]))
    eligible = [r for r in ranked if r[0]>=required_tasks and r[1]>=required_workers]
    if not eligible:
        return dict(status='insufficient_source_density',radius_m=radius_m,
            requested_tasks=required_tasks,requested_workers=required_workers,
            top_counts=[dict(tasks=r[0],workers=r[1],cell=list(r[2])) for r in ranked[:10]])
    count,worker_count,cell,worker_ids = eligible[0]
    center = tuple((x+.5)*radius_m for x in cell)
    task_ids = [key for key,pickup,delivery in requests if math.dist(center,pickup)<=radius_m and math.dist(center,delivery)<=radius_m]
    return dict(status='selected',radius_m=radius_m,center_xy_m=center,task_ids=task_ids,worker_ids=worker_ids,
        eligible_tasks=count,eligible_workers=worker_count,
        rule='both original service endpoints and observed worker positions within the fixed-radius district; maximize source task count, worker count, stable cell ID')


def choose_source_depot_district(requests, depots, *, radius_m=900., required_tasks=108):

    from scipy.spatial import cKDTree
    requests,depots = list(requests),sorted(depots)
    tree = cKDTree([xy for key,xy in depots])
    counts = Counter()
    for key,pickup,delivery in requests:
        for index in tree.query_ball_point(pickup,radius_m):
            if math.dist(depots[index][1],delivery) <= radius_m:
                counts[index] += 1
    ranked = sorted(counts,key=lambda i:(-counts[i],depots[i][0]))
    if not ranked or counts[ranked[0]] < required_tasks:
        return dict(status='insufficient_source_density',radius_m=radius_m,requested_tasks=required_tasks,
            top_counts=[dict(tasks=counts[i],source_depot_id=depots[i][0]) for i in ranked[:10]])
    index = ranked[0]
    center = depots[index][1]
    return dict(status='selected',radius_m=radius_m,center_xy_m=center,source_depot_id=depots[index][0],
        eligible_tasks=counts[index],task_ids=[key for key,pickup,delivery in requests
            if math.dist(center,pickup)<=radius_m and math.dist(center,delivery)<=radius_m],
        rule='both original service endpoints within the fixed radius of an original source depot; maximize source task count, stable depot ID')
