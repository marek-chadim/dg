% oct2mat


function [] = run()


global c

% Model Parameters

c.MAX_FIRMS = 3;
c.START_FIRMS = 1;
c.EQL_TYPE = 'COMPETITION'; % COMPETITION|MONOPOLY|PLANNER

c.IND_TYPE = 'COST'; % QUALITY|COST|CAPACITY (untested, do not change)
c.ENTRY_TYPE = 'RAN_ENTRY';
c.ENTRY_LOW = 0.15;
c.ENTRY_HIGH = 0.25;
c.ENTRY_SUNK = 0.2;
c.ENTRY_AT = 4;
c.BETA = 0.925;
c.DELTA = 0.7;
c.SCRAP_VAL = 0.1;
c.INV_MULT = 3;
c.INV_COST = 1;
c.MC = 5;
c.MKT_SIZE = 5;
c.KMAX = 19;
c.WSTAR = 12;
c.INTERCEPT = 3;
c.FIXED_COST = 0.2;
c.GAMMA = 1;
c.TAU = 0.1;
c.PROFIT_DONE = 0;
c.EQL_DONE = 0;
c.ACTIVE_CFG = 'default';

c.PREFIX = acronym(c.EQL_TYPE, c.IND_TYPE);
c.DS_WSTART = [c.ENTRY_AT+2; zeros(c.MAX_FIRMS-1,1)];
c.DS_NSIMX = 10000; % 10000;
c.DS_NSIMW = 100; % 100;
c.DS_NRUNW = 100; % 100;


% Compute static profit:
%
profit;


% Solve dynamic equilibrium:
%
if strcmp(c.EQL_TYPE, 'COMPETITION') eql_ma;
else eql_sa;
  end

% Descriptive Statistics:
% . Print profit and value functions:
%
% dstat;

% . Simulate entry & exit:
%
if strcmp(c.EQL_TYPE, 'COMPETITION') ds_ma;
else dstat; ds_sa;
  end

% . Simulate welfare:
%
if strcmp(c.EQL_TYPE, 'COMPETITION') welf_ma;
else welf_sa;
  end


function [out1] = acronym(et,it)
  disp(['Model: ' et ', investment in ' it]);
  if strcmp(it, 'QUALITY')
    if strcmp(et, 'COMPETITION')  out1 = 'cb';
    elseif strcmp(et, 'MONOPOLY') out1 = 'mb';
    elseif strcmp(et, 'PLANNER')  out1 = 'sb';
      end
  elseif strcmp(it, 'COST')
    if strcmp(et, 'COMPETITION')  out1 = 'cc';
    elseif strcmp(et, 'MONOPOLY') out1 = 'mc';
    elseif strcmp(et, 'PLANNER')  out1 = 'sc';
      end
  elseif strcmp(it, 'CAPACITY')
    if strcmp(et, 'COMPETITION')  out1 = 'cp';
    elseif strcmp(et, 'MONOPOLY') out1 = 'mp';
    elseif strcmp(et, 'PLANNER')  out1 = 'sp';
      end
    end

