import ctypes as ct
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import weakref

from common.evaluation.recovery_plan import RecoveryPlanValidationError


class ScheduleUnavailable(RuntimeError):
    pass


_FLAGS = ('-std=c11', '-O3', '-shared', '-fPIC', '-fno-fast-math',
          '-ffp-contract=off', '-fexcess-precision=standard', '-fno-associative-math')

_SOURCE = r'''
#include <stdint.h>
#include <stdlib.h>
#include <stddef.h>
#include <math.h>
#include <float.h>
#include <fenv.h>
#include <string.h>
#ifdef __FAST_MATH__
#error Fast math is incompatible with the recovery timing contract.
#endif

typedef struct {
    int32_t kind, key, index, delivery_index, has_ready;
    double duration, ready, promise, service;
} Row;
typedef struct {
    int32_t count;
    const Row *rows;
    double departure;
    double *waiting, *tardiness;
} Route;
typedef struct { int32_t code, route, row, transfer_key; } Error;
typedef struct {
    int32_t kind, key, index, first, second, pred1, pred2, degree, successor;
    double completion;
} Node;
typedef struct {
    const Row *fact;
    int32_t route, index, node, previous;
    double arrival;
} Visit;
typedef struct { int32_t target, next; } Edge;
typedef struct {
    size_t capacity, hash_capacity;
    Node *nodes;
    Visit *visits;
    Edge *edges;
    int32_t *heap, *hash;
} Workspace;

int drt_supported(void) {
    return sizeof(double)==8 && DBL_MANT_DIG==53 && FLT_RADIX==2 &&
           DBL_MAX_EXP==1024 && fegetround()==FE_TONEAREST;
}
int drt_advance(double start, double duration, double *result) {
    double completion=start+duration;
    if (!isfinite(completion)) return 1;
    if (duration>0.0 && completion<=start) {
        double following=nextafter(start, INFINITY);
        if (!isfinite(following) || following-start>1e-9) return 2;
        completion=following;
    }
    *result=completion;
    return 0;
}
void *drt_create(void) { return calloc(1, sizeof(Workspace)); }
void drt_destroy(void *opaque) {
    Workspace *w=(Workspace *)opaque;
    if (!w) return;
    free(w->nodes); free(w->visits); free(w->edges);
    free(w->heap); free(w->hash); free(w);
}
static int reserve(Workspace *w, size_t count) {
    if (count<=w->capacity) return 1;
    size_t capacity=64, hash_capacity;
    while (capacity<count) capacity*=2;
    hash_capacity=capacity*2;
    Node *nodes=malloc(capacity*sizeof(Node));
    Visit *visits=malloc(capacity*sizeof(Visit));
    Edge *edges=malloc(capacity*sizeof(Edge));
    int32_t *heap=malloc(capacity*sizeof(int32_t));
    int32_t *hash=malloc(hash_capacity*sizeof(int32_t));
    if (!nodes || !visits || !edges || !heap || !hash) {
        free(nodes); free(visits); free(edges); free(heap); free(hash);
        return 0;
    }
    free(w->nodes); free(w->visits); free(w->edges);
    free(w->heap); free(w->hash);
    w->nodes=nodes; w->visits=visits; w->edges=edges;
    w->heap=heap; w->hash=hash; w->capacity=capacity; w->hash_capacity=hash_capacity;
    return 1;
}
static int less(const Node *a, const Node *b) {
    if (a->kind!=b->kind) return a->kind<b->kind;
    if (a->key!=b->key) return a->key<b->key;
    return a->index<b->index;
}
static void push(Workspace *w, int32_t *size, int32_t node) {
    int32_t i=(*size)++;
    while (i) {
        int32_t parent=(i-1)/2;
        if (!less(&w->nodes[node], &w->nodes[w->heap[parent]])) break;
        w->heap[i]=w->heap[parent]; i=parent;
    }
    w->heap[i]=node;
}
static int32_t pop(Workspace *w, int32_t *size) {
    int32_t result=w->heap[0], last=w->heap[--(*size)], i=0;
    while (i*2+1<*size) {
        int32_t child=i*2+1;
        if (child+1<*size && less(&w->nodes[w->heap[child+1]], &w->nodes[w->heap[child]])) child++;
        if (!less(&w->nodes[w->heap[child]], &w->nodes[last])) break;
        w->heap[i]=w->heap[child]; i=child;
    }
    if (*size) w->heap[i]=last;
    return result;
}
static int fail(Error *error, int code, int route, int row, int transfer) {
    error->code=code; error->route=route; error->row=row; error->transfer_key=transfer;
    return code;
}

int drt_schedule(void *opaque, int32_t route_count, const Route *const *routes,
                 double transfer_duration, Error *error) {
    Workspace *w=(Workspace *)opaque;
    size_t count=0;
    int32_t node_count=0, visit_count=0, edge_count=0, heap_size=0, completed=0;
    if (!drt_supported()) return fail(error, 9, -1, -1, -1);
    for (int32_t r=0; r<route_count; r++) count+=(size_t)routes[r]->count;
    if (count>INT32_MAX || !reserve(w,count)) return fail(error,9,-1,-1,-1);
    if (!count) return 0;
    memset(w->hash, 0xff, w->hash_capacity*sizeof(int32_t));
    for (int32_t r=0; r<route_count; r++) {
        const Route *route=routes[r];
        int32_t previous=-1;
        for (int32_t i=0; i<route->count; i++) {
            const Row *row=&route->rows[i];
            int32_t node=-1;
            if (row->kind) {
                size_t slot=((uint32_t)row->key*2654435761u)&(w->hash_capacity-1);
                while (w->hash[slot]>=0 && w->nodes[w->hash[slot]].key!=row->key)
                    slot=(slot+1)&(w->hash_capacity-1);
                node=w->hash[slot];
                if (node<0) {
                    w->hash[slot]=node=node_count++;
                    w->nodes[node].key=row->key;
                }
            } else node=node_count++;
            Visit *v=&w->visits[visit_count];
            v->fact=row; v->route=r; v->index=i; v->node=node; v->previous=previous;
            previous=node;
            visit_count++;
        }
    }


    for (int32_t n=0; n<node_count; n++) {
        w->nodes[n].first=w->nodes[n].second=-1;
        w->nodes[n].pred1=w->nodes[n].pred2=-1;
        w->nodes[n].degree=0; w->nodes[n].successor=-1;
    }
    for (int32_t i=0; i<visit_count; i++) {
        Visit *v=&w->visits[i]; Node *n=&w->nodes[v->node];
        if (n->first<0) {
            n->kind=v->fact->kind; n->key=v->fact->key; n->index=v->fact->index; n->first=i;
        } else if (n->second<0 && n->kind) n->second=i;
        else return fail(error,8,v->route,v->index,-1);
        if (v->previous>=0 && n->pred1!=v->previous && n->pred2!=v->previous) {
            if (n->pred1<0) n->pred1=v->previous;
            else if (n->pred2<0) n->pred2=v->previous;
            else return fail(error,8,v->route,v->index,-1);
            n->degree++;
            Edge *edge=&w->edges[edge_count];
            edge->target=v->node; edge->next=w->nodes[v->previous].successor;
            w->nodes[v->previous].successor=edge_count++;
        }
    }
    for (int32_t n=0; n<node_count; n++) {
        if (w->nodes[n].kind && w->nodes[n].second<0) return fail(error,8,-1,-1,w->nodes[n].key);
        if (!w->nodes[n].degree) push(w,&heap_size,n);
    }
    while (heap_size) {
        int32_t id=pop(w,&heap_size); Node *node=&w->nodes[id];
        int32_t ids[2]={node->first,node->second};
        double start=0., completion;
        for (int j=0; j<2 && ids[j]>=0; j++) {
            Visit *v=&w->visits[ids[j]];
            double previous=(v->previous<0 ? routes[v->route]->departure : w->nodes[v->previous].completion);
            int status=drt_advance(previous,v->fact->duration,&v->arrival);
            if (status) return fail(error,status,v->route,v->index,-1);
            if (!j || v->arrival>start) start=v->arrival;
        }
        Visit *first=&w->visits[node->first];
        if (first->fact->has_ready && first->fact->ready>start) start=first->fact->ready;
        int status=drt_advance(start,node->kind ? transfer_duration : first->fact->service,&completion);
        if (status) return fail(error,status+2,first->route,first->index,-1);
        node->completion=completion;
        for (int j=0; j<2 && ids[j]>=0; j++) {
            Visit *v=&w->visits[ids[j]];
            double waiting=start-v->arrival;
            if (!isfinite(waiting)) return fail(error,5,v->route,v->index,-1);
            routes[v->route]->waiting[v->index]=waiting;
        }
        completed++;
        for (int32_t e=node->successor; e>=0; e=w->edges[e].next) {
            int32_t target=w->edges[e].target;
            if (!--w->nodes[target].degree) push(w,&heap_size,target);
        }
    }
    if (completed!=node_count) {
        int32_t transfer=-1;
        for (int32_t n=0; n<node_count; n++)
            if (w->nodes[n].degree && w->nodes[n].kind && (transfer<0 || w->nodes[n].key<transfer)) transfer=w->nodes[n].key;
        return fail(error,7,-1,-1,transfer);
    }

    for (int32_t i=0; i<visit_count; i++) {
        Visit *v=&w->visits[i];
        if (v->fact->delivery_index>=0) {
            double difference=w->nodes[v->node].completion-v->fact->promise;
            double tardiness=(difference>0.0 ? difference : 0.0);
            if (!isfinite(tardiness)) return fail(error,6,v->route,v->index,-1);
            routes[v->route]->tardiness[v->fact->delivery_index]=tardiness;
        }
    }
    return 0;
}
'''


class _CRow(ct.Structure):
    _fields_ = [(name, ct.c_int32) for name in ('kind', 'key', 'index', 'delivery_index', 'has_ready')] + [
        (name, ct.c_double) for name in ('duration', 'ready', 'promise', 'service')]


class _CRoute(ct.Structure):
    _fields_ = [('count', ct.c_int32), ('rows', ct.POINTER(_CRow)), ('departure', ct.c_double),
                ('waiting', ct.POINTER(ct.c_double)), ('tardiness', ct.POINTER(ct.c_double))]


class _CError(ct.Structure):
    _fields_ = [(name, ct.c_int32) for name in ('code', 'route', 'row', 'transfer_key')]


_LIBRARY = None
_LIBRARY_LOCK = threading.Lock()


def _library():
    global _LIBRARY
    with _LIBRARY_LOCK:
        if _LIBRARY is not None:
            return _LIBRARY
        if sys.platform != 'linux' or sys.float_info.mant_dig != 53 or ct.sizeof(ct.c_double) != 8:
            raise ScheduleUnavailable('native scheduling requires Linux and IEEE binary64')
        compiler = shutil.which('gcc')
        if compiler is None:
            raise ScheduleUnavailable('gcc is unavailable; use the Python scheduler')
        try:
            with tempfile.TemporaryDirectory(prefix='drt-sc-native-') as directory:
                source = Path(directory) / 'schedule.c'
                output = Path(directory) / 'schedule.so'
                source.write_text(_SOURCE)
                subprocess.run([compiler, *_FLAGS, str(source), '-o', str(output), '-lm'],
                               check=True, text=True, capture_output=True)
                library = ct.CDLL(str(output))
                library.drt_supported.restype = ct.c_int
                library.drt_create.restype = ct.c_void_p
                library.drt_destroy.argtypes = [ct.c_void_p]
                library.drt_schedule.argtypes = [ct.c_void_p, ct.c_int32, ct.POINTER(ct.POINTER(_CRoute)), ct.c_double, ct.POINTER(_CError)]
                library.drt_schedule.restype = ct.c_int
                library.drt_advance.argtypes = [ct.c_double, ct.c_double, ct.POINTER(ct.c_double)]
                library.drt_advance.restype = ct.c_int
                if not library.drt_supported():
                    raise ScheduleUnavailable('native floating-point environment is unsupported')
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            raise ScheduleUnavailable('native scheduler could not be compiled: '+str(error)) from error
        _LIBRARY = library, dict(compiler=compiler, flags=list(_FLAGS))
        return _LIBRARY


class PreparedRoute:


    def __init__(self, kernel, worker_id, rows, departure):
        self.kernel, self.worker_id, self.rows = kernel, worker_id, tuple(rows)
        facts, delivery_count, previous = [], 0, None
        for index, row in enumerate(self.rows):
            if row.previous != previous or row.context['worker_id'] != worker_id:
                raise ScheduleUnavailable('route template does not preserve native visit order')
            if row.node == ('service', worker_id, index):
                kind, key = 0, kernel._workers[worker_id]
            elif len(row.node) == 3 and row.node[0] == 'transfer' and row.node[2] == 0:
                kind = 1
                try:
                    key = kernel._transfers[row.node[1]]
                except KeyError as error:
                    raise ScheduleUnavailable('transfer node has no fixed native ordering rank') from error
            else:
                raise ScheduleUnavailable('private/custom timing nodes require Python scheduling')
            delivery = -1 if row.task_id is None else delivery_count
            delivery_count += delivery >= 0
            facts.append(_CRow(kind, key, index if not kind else 0, delivery,
                               row.ready is not None, row.duration,
                               0. if row.ready is None else row.ready,
                               0. if row.promise is None else row.promise, getattr(row, 'service', 0.)))
            previous = row.node
        self._facts = (_CRow * len(facts))(*facts)
        self._waiting = (ct.c_double * len(facts))()
        self._tardiness = (ct.c_double * delivery_count)()
        self._route = _CRoute(len(facts), self._facts, departure, self._waiting, self._tardiness)
        self._pointer = ct.pointer(self._route)
        self.travel = tuple(row.duration for row in self.rows)
        self.distance = tuple(row.distance for row in self.rows)


class ScheduleKernel:
    def __init__(self, worker_ids, transfer_ids, transfer_duration_s):
        self._library, self.receipt = _library()
        self._workers = {w: i for i, w in enumerate(sorted(worker_ids))}
        self._transfer_ids = tuple(sorted(transfer_ids))
        self._transfers = {event: i for i, event in enumerate(self._transfer_ids)}
        self.transfer_duration_s = transfer_duration_s
        self._workspace = self._library.drt_create()
        if not self._workspace:
            raise ScheduleUnavailable('native scheduling workspace allocation failed')
        self._finalizer = weakref.finalize(self, self._library.drt_destroy, self._workspace)
        self._pointers = (ct.POINTER(_CRoute) * len(self._workers))()
        self._error = _CError()
        self._lock = threading.Lock()

    def prepare(self, worker_id, rows, departure_time):
        if worker_id not in self._workers:
            raise ScheduleUnavailable('worker has no fixed native ordering rank')
        return PreparedRoute(self, worker_id, rows, departure_time)

    def run(self, prepared_routes):
        if len(prepared_routes) > len(self._pointers):
            raise ScheduleUnavailable('too many routes for native scheduling workspace')
        with self._lock:
            seen = set()
            for index, route in enumerate(prepared_routes):
                if route.kernel is not self or route.worker_id in seen:
                    raise ScheduleUnavailable('foreign or duplicate native route template')
                seen.add(route.worker_id)
                self._pointers[index] = route._pointer
            status = self._library.drt_schedule(self._workspace, len(prepared_routes), self._pointers,
                                                self.transfer_duration_s, ct.byref(self._error))
            if status:
                self._raise(prepared_routes)
            return {route.worker_id: (tuple(route._tardiness), route.travel,
                                      tuple(route._waiting), route.distance) for route in prepared_routes}

    def _raise(self, routes):
        error = self._error
        if error.code in (8, 9):
            raise ScheduleUnavailable('native scheduling graph/runtime is unsupported')
        if error.code == 7:
            event = self._transfer_ids[error.transfer_key] if error.transfer_key >= 0 else None
            raise RecoveryPlanValidationError('transfer_cycle', 'worker visit orders contain a circular transfer dependency',
                                              transfer_id=event)
        context = routes[error.route].rows[error.row].context
        if error.code in (1, 3, 5, 6):
            name = {1: 'travel', 3: 'transfer duration', 5: 'waiting time', 6: 'tardiness'}[error.code]
            message = name + ' exceeds the finite numeric range'
        else:
            name = 'travel' if error.code == 2 else 'transfer duration'
            message = name + " is below the time base's resolution"
        raise RecoveryPlanValidationError('numeric_range', message, **context)
