function [agents, output, time_step] = grape_MRTA_game_leqL(P, Y, agents, MST_comm, N_timesteps)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% GRAPE Task Allocation_newation Solver Module
% By Inmo Jang, 2.Apr.2016
% Modified, 15.Jul.2016
% Modified, 25.Oct.2017
% Modified for Asynchronous communication environment
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The following describes the name of variables; meanings;  and their matrix sizes
% Input :
%   - n;    the number of agents
%   - m;    the number of tasks
%   - environment.t_location;   Task Position(x,y);             m by 2 matrix (m = #tasks)
%   - environment.t_demand;     Task demand or reward;          m by 1 matrix
%   - environment.a_location;   Agent Posision(x,y);             n by 2 matrix (n = #agents)
%   - Alloc_existing;    Current allocation status of agents;        n by 1 matrix
%   - Flag_display; Flag for display the process;   1 by 1 matrix
% Output :
%   - Alloc;        New allocation status of agents;   n by 1 matrix
%   - a_utility;    Resulted individual utility for each agent; n by 1 matrix
%   - iteration;    Resulted number of iteration for convergence;   1 by 1 matrix

%% Interface (Input)

[~, n] = size(agents);
[~, m] = size(P);
l = agents(1).max_num_tasks;
utility_history = zeros(N_timesteps, n+1);
MST = MST_comm;
G_comm = graph(MST_comm);

environment.t_location = P';
environment.t_demand = 10*ones(1, m);

Alloc_existing = zeros(n, m);
P_r = zeros(n,2);
for i = 1:n
    P_r(i,:) = agents(i).loc;
    Alloc_existing(i,:) = agents(i).assignments;
end 

environment.a_location = P_r;

%% For visualisation
Alloc_history = zeros(n,10);
Satisfied_history = zeros(n,10);
iteration_history = [];


%% Initialisation

a_satisfied = 0; % # Agents who satisfy the current partition

for i=1:n
    agents(i).iteration = 0;
    agents(i).time_stamp = rand;
    agents(i).A = Alloc_existing;
    agents(i).satisfied_flag = 0;
    agents(i).utility= 0;
end

%% Neighbour agents identification (Assumming a static situation)
for i=1:n
    agent_info(i).set_neighbour_agent_id = find(MST(i,:)>0);
end
Iteration_agent_current = zeros(n,1);
Timestamp_agent_current = zeros(n,1);

%% GRAPE Algorithm
time_step = 0;
while a_satisfied~=n && time_step < N_timesteps
    time_step = time_step + 1;

    for i=1:n % For Each Agent          
        %%%%% Line 5 of Algorithm 1
        Alloc_ = agents(i).A;
        current_assignment = Alloc_(i,:); % Currently-selected tasks

        curr_utility = utility_calculation(agents(i), P, Y);
        updated_assignment = current_assignment;
        if sum(current_assignment) < l
            Candidate = ones(m,1)*(-inf);
            for t=1:m
                if current_assignment(t) == 0
                    poss_assignment = current_assignment;
                    poss_assignment(t) = 1;
                    poss_agent = agents(i);
                    poss_agent.A = Alloc_;
                    poss_agent.A(i,:) = poss_assignment;
                    poss_agent.assignments = poss_agent.A(i,:);
                    Candidate(t) = utility_calculation(poss_agent, P, Y);
                end
            end
            % Select Best alternative
            [Best_utility, Best_task_id] = max(Candidate);
            if Best_utility > curr_utility
                updated_assignment(Best_task_id) = 1;           
            end
        end
        %%%%% End of Line 5 of Algorithm 1

        
        
        %%%%% Line 6-11 of Algorithm 1
        Alloc_(i, :) = 0;
        selected_tasks = find(updated_assignment);
        Alloc_(i, selected_tasks) = 1;
        
        if current_assignment == Alloc_(i,:) % if this choice is the same as remaining
            agents(i).satisfied_flag = 1;            
        else
            agents(i).satisfied_flag = 1;
            agents(i).A = Alloc_;
            agents(i).time_stamp = rand;
            agents(i).iteration = agents(i).iteration + 1;
            agents(i).assignments = Alloc_(i,:);
            agents(i).selected_tasks = find(agents(i).assignments);
        end        
        agents(i).utility = utility_calculation(agents(i), P, Y);
        
        %%%%% End of Line 6-11 of Algorithm 1
        
        % For speed up when executing Algorithm 2
        Iteration_agent_current(i) = agents(i).iteration;
        Timestamp_agent_current(i) = agents(i).time_stamp;        
    end

    % bookkeeping
    for a_i = 1:n
        poss_agent = agents(a_i);
        for a_l = 1:n
            poss_agent.A(a_l,:) = agents(a_l).A(a_l,:);
        end
        agent_utility = utility_calculation(poss_agent, P, Y);
        utility_history(time_step, a_i) = agent_utility;
    end
    utility_history(time_step, end) = sum(utility_history(time_step, 1:end-1));

    %% Distributed Mutex (Algorithm 2)  

    agents_ = agents;
    for i=1:n
        set_neighbour_agent_id = find(MST(i,:)>0);
        % Initially
        agents_(i).satisfied_flag = 1;
        agents_(i).A = agents(i).A;
        agents_(i).time_stamp = agents(i).time_stamp;
        agents_(i).iteration = agents(i).iteration;
        agents_(i).utility= agents(i).utility;
        agents_(i).completion_time_vec = agents(i).completion_time_vec;

%       (Revision) To find out the local "deciding agent" amongst neighbour agents
        set_neighbour_agent_id_ = [set_neighbour_agent_id i];
        % Iteratation amongst neighbour agent set
        Iteration_agent_neighbour = Iteration_agent_current(set_neighbour_agent_id_);
        % Maximum iteration amongst neighbour agent set
        max_Iteration = max(Iteration_agent_neighbour);
        % Agents who have maximum iteration
        max_Iteration_agent_neighbour = (Iteration_agent_neighbour == max_Iteration);
        
        % Timestamp amongst neighbour agent set
        Timestamp_agent_neighbour = Timestamp_agent_current(set_neighbour_agent_id_);
        % Time stamps amongst neighbour agent who have maximum iteraiton
        Timestamp_agent_maxiteration = Timestamp_agent_neighbour.*max_Iteration_agent_neighbour;
        
        [max_Timestamp, agent_neighbour_index] = max(Timestamp_agent_maxiteration);
        valid_agent_id = set_neighbour_agent_id_(agent_neighbour_index);  % Find out "deciding agent" 
        
        % Update local information from the deciding agent's local information
        agents_(i).A = agents(valid_agent_id).A;
        agents_(i).time_stamp = agents(valid_agent_id).time_stamp;
        agents_(i).iteration = agents(valid_agent_id).iteration;
        agents_(i).assignments = agents_(i).A(i,:);
        agents_(i).selected_tasks = find(agents_(i).assignments);        
        agents_(i).completion_time_vec = agents(valid_agent_id).completion_time_vec;

        if min(agents(i).A == agents_(i).A, [], 'all')==1  && all(agents_(i).completion_time_vec == agents(i).completion_time_vec)% If local information is changed
        else
            agents_(i).satisfied_flag = 0;
        end
    end    
    agents = agents_;
    
    %% Check the current status
    a_satisfied = 0;
    iteration = 1;
    for i=1:n
        if agents(i).satisfied_flag == 1
        a_satisfied = a_satisfied + 1;
        end
        % Check the maximum iteration
        iteration = max(agents(i).iteration,iteration);
    end
end

%% Last Check: If Alloc is consensused?
a_utility = zeros(n,1);
output.flag_problem = 0;
for i=1:n
    if i==1
        Alloc_1 = agents(i).A;
        iteration_1 = agents(i).iteration;
        time_stamp_1 = agents(i).time_stamp;
        agents(i).assignemnt =  agents(i).A(i,:);
        agents(i).selected_tasks = find(agents(i).assignemnt);
        agents(i).utility = utility_calculation(agents(i), P, Y);
    else
        Alloc = agents(i).A;        
        iteration = agents(i).iteration;
        time_stamp = agents(i).time_stamp;
        
        if (all(Alloc_1 == Alloc, 'all'))&&(iteration_1 == iteration)&&(time_stamp_1 == time_stamp)
            % Consensus OK
        else
            disp(['Problem: Non Consensus with Agent#1 and Agent#',num2str(i)]);
            output.flag_problem = 1;
        end        
    end
    a_utility(i) = agents(i).utility;
end

%% Interface (Output)

output.A = Alloc;
output.a_utility = a_utility;
output.iteration = iteration;
output.utility_history = utility_history;  

output.visual.A_history = Alloc_history;
output.visual.Satisfied_history = Satisfied_history;
output.visual.iteration_history = iteration_history;


end