from builtins import min

import scipy.spatial as sp
import pandas as pd
import numpy as np
import logging as log
import json
import sys
# also required packages: openpyxl
from ortools.linear_solver import pywraplp

def solve_OrTools(dima):
    '''
    generate mip model using google or-tools and solve it

    :param dima: the distance matrix
    :return:  solution X, model, status
    '''

    # Create the model.
    solver_name = 'GUROBI_MIP'
    log.info('instantiating solver ' + solver_name)
    model = pywraplp.Solver.CreateSolver(solver_name)
    model.EnableOutput()

    # log.info('Defining MIP model... ')
    # # generating decision variables X_ij
    # log.info('Creating ' + str(len(all_services) * len(all_services)) + ' boolean X_ij variables... ')
    # X = {}
    # for i in all_services:
    #     for j in all_services:
    #         X[(i, j)] = model.BoolVar('X_i%ij%i' % (i, j))
    #
    # # generating help variables LF_ij (for Leerfahrten)
    # log.info('Creating ' + str(len(all_services) * len(all_services)) + ' boolean LF_ij variables... ')
    # LF = {}
    # for i in all_services:
    #     for j in all_services:
    #         LF[(i, j)] = model.BoolVar('LF_i%ij%i' % (i, j))
    #
    # # generating help variables u_i (for subtour elimination)
    # log.info('Creating ' + str(len(all_services)) + ' boolean u_i variables... ')
    # u = {}
    # for i in all_services:
    #     u[i] = model.IntVar(0, mpar.size, 'u_i%i' % i)
    #
    # # constraint 1: accessibility and empty runs
    # log.info('Creating ' + str(len(all_services) * len(all_services)) + ' Constraint 1... ')
    # for i in all_services:
    #     for j in all_services:
    #         log.debug('Constraint 1, (i,j)=(' + str(i) + ', ' + str(j))
    #         model.Add(X[(i, j)] * (1 - s[(i, j)]) <= LF[(i, j)])
    #         # model.Add(X[(i, j)] <= s[(i, j)])
    #
    # # constraint 3.1: fake start-service has no predecessor
    # log.info('Creating ' + str(len(fake_start_services)) + ' Constraint 3.1... ')
    # for j in fake_start_services:
    #     model.Add(sum(X[(i, j)] for i in all_services) == 0)
    #
    # # constraint 3.2: fake end-service has no successor
    # log.info('Creating ' + str(len(fake_end_services)) + ' Constraint 3.2... ')
    # for i in fake_end_services:
    #     model.Add(sum(X[(i, j)] for j in all_services) == 0)
    #
    # # constraint 4.1: all non-fake end services must be served exactly once
    # log.info('Creating ' + str(len(all_services)) + ' Constraint 4.1... ')
    # for i in all_services_non_fake_end:
    #     model.Add(sum(X[(i, j)] for j in all_services) == 1)
    #
    # # constraint 4.2: all non-fake start service must be served exactly once
    # log.info('Creating ' + str(len(fake_end_services)) + ' Constraint 4.2... ')
    # for j in all_services_non_fake_start:
    #     model.Add(sum(X[(i, j)] for i in all_services) == 1)
    #
    # # constraint 4.3: fake-services may not be connected
    # log.info('Creating ' + str(len(fake_end_services)) + ' Constraint 4.3... ')
    # model.Add(X[(fake_start_services[0], fake_end_services[0])] == 0)
    #
    # # constraint 6: the plan must have exactly the specified duration
    # log.info('Creating 1 Constraint 6... ')
    # model.Add(sum(X[(i, j)] * p[(i, j)] for i in all_services for j in all_services) == mpar.tc_num_weeks * 7 + 1)
    #
    # # constraint 7.1: subtour elimination constraints (Miller-Tucker-Zemlin) part 1
    # log.info('Creating 1 Constraint 7.1... ')
    # model.Add(u[0] == 1)
    #
    # # constraint 7.2: subtour elimination constraints (Miller-Tucker-Zemlin) part 2
    # log.info('Creating ' + str(len(all_services_non_zero)) + ' Constraint 7.2... ')
    # for i in all_services_non_zero:
    #     model.Add(2 <= u[i])
    #     # TODO: check < or <= ? should be <, I suppose
    #     model.Add(u[i] <= mpar.size)
    #
    # # constraint 7.3: subtour elimination constraints (Miller-Tucker-Zemlin) part 3
    # log.info('Creating ' + str(len(all_services_non_zero)) + ' Constraint 7.2... ')
    # for i in all_services_non_zero:
    #     for j in all_services_non_zero:
    #         model.Add(u[i] - u[j] + 1 <= (mpar.size - 1)*(1 - X[(i, j)]))
    #
    # # Minimierung der Anzahl geänderte Dienste und Anzahl Dienstfahrten
    # model.Minimize(
    #     weight_alpha_1 * sum(LF[(i, j)] for i in all_services for j in all_services))
    #
    # log.info('Solving MIP model... ')
    # status = model.Solve()
    #
    return X, model, status

def get_services_dict(X, LF, num_services, fake_start_rowid, fake_end_rowid):
    '''
    create a map from service-row-id to the 0-based index of service for all services (containing fake start-/end-svcs)

    :param X: the decision variable X_ij
    :param LF: the decision variable LF_ij
    :param num_services: the number of services
    :param fake_start_rowid: the rowid of the only fake start service
    :param fake_end_rowid: the rowid of the only fake end service
    :return: a map from service-row-id to the 0-based index of service for all services (containing fake start-/end-svcs)
             as well as a list containing the start service-row-id of all (i,j) tupels requiring a Leerfahrt
    '''
    # first, create a map from each row-id to successor row-id
    successor_map = {}
    lf_list = []
    all_services = range(0, num_services)
    for i in all_services:
        for j in all_services:
            # this following check should be executed conditionally if X_ij = 1, but for reasons of better
            # error-detection / transparency we check this here independently!
            if LF[(i, j)].solution_value() == 1:
                lf_list.append(i)
                # log.info('LF ' + str(i) + '/' + str(j))
            if X[(i, j)].solution_value() == 1:
                successor_map[i] = j
                # log.info('Service ' + str(i) + '/' + str(j))
                break
    # now create the result map
    next_row_id = fake_start_rowid
    row_id = 0
    sequential_rowid_dict = {}
    lf_list_trans = []
    while next_row_id in successor_map:
        sequential_rowid_dict[next_row_id] = row_id
        if next_row_id in lf_list:
            lf_list_trans.append(row_id)
        next_row_id = successor_map[next_row_id]
        row_id = row_id + 1
    return sequential_rowid_dict, lf_list_trans

def main():
    # configure logger for info level
    log.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=log.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        stream=sys.stdout)

    # load tsp instance
    tsp_problem = 'qa194.tsp'

    log.info("Reading TSP problem Instance " + tsp_problem)
    tsp = pd.read_csv('./TSP_Instances/' + tsp_problem, sep=' ', skiprows=7, dtype = float,
                      names=['nodeId', 'lat', 'lng'], skipfooter=1, engine='python')
    tsp = tsp.sort_values(by='nodeId', inplace=False)

    A = tsp[['lat', 'lng']].to_numpy()
    dima = sp.distance_matrix(A, A)

    # now solve problem
    X, model, status = solve_OrTools(dima)

    # check problem response
    if status == pywraplp.Solver.OPTIMAL:
        log.info('Solution:')
        log.info('optimal solution found.')
        log.info('Objective value =' + str(model.Objective().Value()))
    elif status == pywraplp.Solver.INFEASIBLE:
        log.info('The problem is infeasible.')
    else:
        log.info('The problem could not be solved. Return state was: ' + status)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
