#!/usr/bin/env python

"""
FSA module
This module implements simple deterministic and non-deterministic finite 
state automata. It follows the description in Chapter 2 of Jurafsky and 
Martin (2008).
@author: Rob Malouf
@organization: Dept. of Linguistics, San Diego State University
@contact: rmalouf@mail.sdsu.edu
@version: 4
@date: 8-April-2015
@since: 30-January-2008
"""

from copy import deepcopy

class FSA(object):

    """
    Generic finite-state automaton class
    
    An FSA consists of (see pg. 28):
      - an alphabet (C{self.Sigma})
      - an initial state (C{self.q0})
      - a set of final states (C{self.F})
      - a transition function (C{self.delta})
      
    """

    def __init__(self, delta, F, q0=0):
  
        """
        Create a new FSA object
                
        @param delta: the transition function
        @type delta: a C{dict} of C{dict}s
        @param F: final states
        @type F: C{list}
        @param q0: initial state (by default, 0)
        @type q0: C{int}
        @raise ValueError: the FSA is mis-specified somehow
        """

        # check transition function
        for state in delta:
            for symbol in delta[state]:
                # the value of the transition function should be a list of states
                if not isinstance(delta[state][symbol], list):
                    delta[state][symbol] = [ delta[state][symbol] ]
                # verify that each transition leads to a real state
                for state1 in delta[state][symbol]:
                    if state1 not in delta:
                        raise ValueError, '%s not a valid state'%(state1)
        self.delta = delta

        # check initial state
        if q0 not in self.delta:
            raise ValueError, '%s not a valid initial state'%(q0)
        self.q0 = q0
        
        # check final states
        for final in F:
            if final not in self.delta:
                raise ValueError, '%s is not a valid final state'%(final)
        self.F = set(F)

        # collect symbols from transitions to construct the alphabet
        self.Sigma = set()
        for trans in self.delta.values():
            self.Sigma.update(trans.keys())

    def __repr__(self):
        """Produce a string representation of an FSA (e.g., for printing)."""

        result = '{ %s\n' % (self.__class__.__name__)
        result += '  Alphabet : %s\n'%self.Sigma
        for state in self.delta:
            result += '  %s:' % state
            if state == self.q0:
                result += ' initial'
            if state in self.F:
                result += ' final'
            result += '\n'
            for sym in self.delta[state]:
                if hasattr(self, 'weights'):
                    # weighted FSAs
                    for state1,weight in zip(self.delta[state][sym], self.weights[state][sym]):
                        result += '    %s => %s/%.2f\n'%(sym, state1, weight)
                else:
                    # unweighted FSAs                       
                    for state1 in self.delta[state][sym]:
                        result += '    %s => %s\n'%(sym, state1)       
        result += '}'
        return result


    def visit(self, trace, index, state):
        """Trace: visit a node in an FSA"""

        if trace:
            print 'Obs %d at state %d'%(index,state)

    def accept(self, trace):
        """Trace: FSA accepts a string"""

        if trace:
            print 'ACCEPT'

    def reject(self, trace, reason=None):
        """Trace: FSA rejects a string"""

        if trace:
            if reason:
                print 'REJECT (%s)'%reason
            else:
                print 'REJECT'

    
class DFSA(FSA):
    """Deterministic FSA class"""

    def __init__(self, *args, **kwargs):

        # create FSA
        super(DFSA, self).__init__(*args, **kwargs)
        
        # verify that it is deterministic
        if '' in self.Sigma:
            raise ValueError, 'epsilon transitions not allowed in DFSAs'
        for state in self.delta:
            for state1 in self.delta[state].values():
                if len(state1) > 1:
                    raise ValueError, 'non-deterministic transitions not allowed in DFSAs'

    def complement(self):
        """return the complement a DFSA"""
        
        # copy delta
        compl = deepcopy(self.delta)
        
        # add sink state...
        compl['sink'] = { }

        # make complete...
        for state in compl:
            for sym in self.Sigma:
                if sym not in compl[state]:
                    compl[state][sym] = 'sink'

        # final states
        F = set(state for state in compl if state not in self.F)

        return DFSA(compl, F)                                

    def recognize(self, tape, trace=False):
        """check whether an input tape is accepted by the FSA (see fig. 2.12)"""

        index = 0
        current_state = self.q0

        while True:

            self.visit(trace, index, current_state)

            if index == len(tape):

                # end of input tape
                if current_state in self.F:
                    self.accept(trace)
                    return True
                else:
                    self.reject(trace, '%s is not an accepting state'%current_state)
                    return False

            else:

                if tape[index] in self.delta[current_state]:
                    # follow transition to next state                   
                    current_state = self.delta[current_state][tape[index]][0]
                    index += 1
                else:
                    # no transition!
                    self.reject(trace, 'no transition for "%s" at state %s'%(tape[index], current_state))
                    return False

class NFSA(FSA):
    """Non-deterministic FSA class"""

    def __init__(self, *args, **kwargs):

        # create FSA
        super(NFSA, self).__init__(*args, **kwargs)    

        # check for epsilon loops by DFS
        states = self.delta.keys()
        while states:
            agenda = [ states[0] ]
            visited = set()
            while agenda:
                state = agenda.pop(0)
                if state in visited:            
                    raise ValueError,'epsilon loop!'
                elif state in states:
                    states.remove(state)
                    visited.add(state)
                    if '' in self.delta[state]:
                        for j in self.delta[state]['']:
                            agenda.append(j)

    def recognize(self, tape, trace=False, depth_first=True):
        """check whether an input string is accepted by the FSA (see fig. 2.19)"""

        agenda = [ (self.q0, 0) ]

        while True:
        
            if agenda:
                if depth_first:
                    # depth first search
                    (node, index) = agenda.pop()
                else:
                    # breadth first search
                    (node, index) = agenda.pop(0)
            else:
                self.reject(trace, 'no more agenda items')
                return False
            
            self.visit(trace, index, node)

            # accepting state?
            if index == len(tape) and node in self.F:
                # yes, so accept
                self.accept(trace)
                return True
            else:
                # generate new states
                if '' in self.delta[node]:
                    for new_node in self.delta[node]['']:
                        # follow epsilon transitions
                        agenda.append((new_node, index))
                if index < len(tape) and tape[index] in self.delta[node]:
                    for new_node in self.delta[node][tape[index]]:
                        agenda.append((new_node, index+1))

class WFSA(NFSA):
    """Weighted non-deterministic FSA class"""

    def __init__(self, delta, F, weights, **kwargs):

        # create unweighted NFSA
        super(WFSA,self).__init__(delta, F, **kwargs)

        # check weight table -- the weight table should have exactly the same form as the 
        # transition table, but with transition weights instead of destination states
        # as the values
        
        assert set(weights.keys()) == set(self.delta.keys())

        for state in weights:
            # make sure we have the same transitions in both tables
            if set(self.delta[state].keys()) != set(weights[state].keys()):
                raise ValueError, "transition table and weight table don't match!"
            for symbol in weights[state]:
                # the value of the weight function should be a list of weights, with as 
                # many elements as there are state in the transition function
                if not isinstance(weights[state][symbol], list):
                    weights[state][symbol] = [ weights[state][symbol] ]     
                if len(self.delta[state][symbol]) != len(weights[state][symbol]):
                    raise ValueError, "transition table and weight table don't match!"
        self.weights = weights

    def cost(self, states, symbols):
        """Find the cost of a path through a WFSA"""
        
        assert len(states) == len(symbols)+1

        c = 0.0
        state = self.q0

        try:
            for i in xrange(1, len(states)):
                # find the transition that goes from the previous state to the next state
                j = self.delta[states[i-1]][symbols[i-1]].index(states[i])
                # add the cost of the transition to the cost of the path
                c += self.weights[states[i-1]][symbols[i-1]][j]
        except KeyError:
            raise KeyError, 'invalid state!'
        except ValueError:
            raise ValueError, 'invalid symbol!'
        return c


### some examples

        
# The sheep language machine (Fig 2.12, J & M)
FSA2_12 = DFSA( { 0: { 'b': 1 },           # state 0  b => 1
                  1: { 'a' : 2 },          # state 1  a => 2
                  2: { 'a' : 3 },          # state 2  a => 3
                  3: { 'a' : 3, '!' : 4},  # state 3  a => 3,  ! => 4
                  4: {  } },               # state 4  
                  [ 4 ])
    
# Fig. 2.18
FSA2_18 = NFSA( { 0: { 'b' : 1 },
                  1: { 'a' : 2 },
                  2: { 'a' : [ 2, 3 ] },
                  3: { '!' : 4 },
                  4: { } },
                  [ 4 ] )
    
# Fig. 2.19
FSA2_19 = NFSA( { 0: { 'b' : 1 },
                  1: { 'a' : 2 },
                  2: { 'a' : 3 },
                  3: { ''  : 2,
                       '!' : 4 },
                  4: { } },
                  [ 4 ] )

# An example WFSA, with a weight of .1 on each transition
wfsa1 = WFSA( {0: { 'b' : 1 },
               1: { 'a' : 2 },
               2: { 'a' : 3 },
               3: { ''  : 2,
                    '!' : 4 },
               4: { } },
             [ 4 ],
             weights =
             { 0: { 'b' : .1 },
               1: { 'a' : .1 },
               2: { 'a' : .1 },
               3: { ''  : .1,
                    '!' : .1 },
               4:{ } },
             )

wfsa2 = WFSA( { 0: { 'b' : 1 },
               1: { 'a' : 2 },
               2: { 'a' : [ 2, 3 ] },
               3: { '!' : 4 },
               4: { } },
             [ 4 ],
             weights =
             { 0: { 'b' : .1 },
               1: { 'a' : .1 },
               2: { 'a' : [ .1, .1 ] },
               3: { '!' : .1 },
               4: { } }
             )

# answer to the question 3b. cost of 0-1-2-1
wfsa3 = WFSA ( {0: {'b': [2,4], 'a' : 1 },
               1: { 'b' : [2,3] },
               2: { 'a' : 1 },
               3: { 'a' : 0},
               4: { 'a' : 0 } },
             [ 0,1 ],
             
             weights =
             {0: {'b': [5,1], 'a' : 2 },
               1: { 'b' : [2,1] },
               2: { 'a' : 2 },
               3: { 'a' : 1},
               4: { 'a' : 6 } }
             )

cost = wfsa3.cost([0,1,2,1],['a', 'b', 'a'])

print cost 

# question 3a. Verification that the shortes path equals 8
wfsa4 = WFSA ( {0: {'b': [2,4], 'a' : 1 },
               1: { 'b' : [2,3] },
               2: { 'a' : 1 },
               3: { 'a' : 0},
               4: { 'a' : 0 } },
             [ 0,1 ],
             
             weights =
             {0: {'b': [5,1], 'a' : 2 },
               1: { 'b' : [2,1] },
               2: { 'a' : 2 },
               3: { 'a' : 1},
               4: { 'a' : 6 } }
             )
cost = wfsa4.cost([0,1,2,1, 3, 0], ['a', 'b', 'a','b', 'a'])
#  print cost


"""@var: Another example WFSA"""