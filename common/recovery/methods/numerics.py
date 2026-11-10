from fractions import Fraction as F
import sys

from common.data.roads.routing import Routing


_U = F(1, 2**53)
_Q = F(1, 2**1074)


def public_cost_underestimate_bound(engine, visit_count, public_cost):


    if (type(engine.routing) is not Routing or sys.float_info.radix != 2
            or sys.float_info.mant_dig != 53 or sys.float_info.rounds != 1
            or engine.item.time_s < 0):
        return None
    u, q = _U, _Q
    upper = F(public_cost)
    promises = tuple(F(task.promised_delivery_time_s) for task in engine.item.tasks)
    if upper < 0 or not promises:
        return None


    clock = max(F(1), F(engine.item.time_s), max(F(0), max(promises))+4*upper+4)
    promise_size = max(abs(value) for value in promises)
    legs = len(engine.item.roads.nodes)+2
    route_error = (16*u*clock+2*legs*q)/(1-8*u)
    clock_error = visit_count*(route_error+6*u*clock+2*q)
    return len(promises)*(clock_error+u*(clock+promise_size)+q)+2*u*(2*upper+1)+q


def public_improvement_bound(public_cost, exact_lower_bound, underestimate):

    difference = max(F(0), F(public_cost)-F(exact_lower_bound))+underestimate
    return (1+_U)*difference+_Q
