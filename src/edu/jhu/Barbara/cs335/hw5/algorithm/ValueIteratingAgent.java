package edu.jhu.Barbara.cs335.hw5.algorithm;

import edu.jhu.Barbara.cs335.hw5.data.Action;
import edu.jhu.Barbara.cs335.hw5.data.State;
import edu.jhu.Barbara.cs335.hw5.data.Terrain;
import edu.jhu.Barbara.cs335.hw5.data.WorldMap;
import edu.jhu.Barbara.cs335.hw5.util.DefaultValueHashMap;
import edu.jhu.Barbara.cs335.hw5.util.Pair;

import java.util.HashSet;
import java.util.Map;
import java.util.Random;
import java.util.Set;

public class ValueIteratingAgent implements ReinforcementLearningAgent
{
	private static final long serialVersionUID = 1L;
	
	/** A mapping between states in that world and their expected values. */
	private Map<State, Double> expectedValues;
	
	/** The world in which this agent is operating. */
	private WorldMap world;	
	/** The discount factor for this agent. */
	private double discountFactor;
	/** The transition function that this agent uses. */
	private TransitionFunction transitionFunction;
	/** The reward function that this agent uses. */
	private RewardFunction rewardFunction;
	/** The convergence tolerance (epsilon). */
	private double convergenceTolerance;

	/**
	 * Creates a new value iterating agent.
	 * @param world The world in which the agent will learn.
	 */
	public ValueIteratingAgent()
	{
		this.expectedValues = new DefaultValueHashMap<>(0.0);
		this.world = null;
		this.discountFactor = 0.5;
		this.transitionFunction = null;
		this.rewardFunction = null;
		this.convergenceTolerance = 0.000000001;
	}

	@Override
	public Policy getPolicy()
	{
		return new ValuePolicy();
	}

	/**
	 * Iterate performs a single update of the estimated utilities of each
	 * state.  Return value specifies whether a termination criterion has been
	 * met.
	 */
	@Override
	public boolean iterate()
	{
		// TODO: implement value iteration; this is basically the inside of the
		// while(!done) loop.

		double delta = 0.0; //Maximum convergence
		/** Traverses all possible states of the map: */
		int width = world.getSize().getFirst();
		int height = world.getSize().getSecond();
		for(int i = 0; i < width; i++) {
			for (int j = 0; j < height; j++) {
				Pair<Integer, Integer> position = new Pair<>(i,j);
				if (world.getTerrain(position) != Terrain.WALL) {
					for (Action action : Action.LEGAL_ACTIONS) {
						/** Since the maximum velocity is capped at 5, we can determine all possible velocities, in
						 *  order to survey all possible states:
						 */
						for(int k = -5; k <=5; k++){
							for(int l = -5; l <= 5; l++){
								Pair<Integer, Integer> velocity =
										new Pair<>(k, l);
								State state = new State(position, velocity);
								/**Calculate the REWARD given at each state: */
								Double reward = rewardFunction.reward(state);
								/**Calculate the MAX UTILITY of each state: */
								Double maxUtility = Double.MIN_VALUE;
								Action maxAction = null;
								for(Action actionPrime : Action.LEGAL_ACTIONS) {
									/** The transition model function returns a set of all possible actions following state s, along with
									 *  their probability of occurring: */
									Double currentUtility = 0.0;
									for(Pair<State, Double> sPrime : transitionFunction.transition(state, actionPrime)) {
										/** The Utility, U, of each state, multiplied by its probability of occurring:  */
										Double probabilitySPrime = sPrime.getSecond();
										Double utilitySPrime = expectedValues.get(sPrime.getFirst());
										currentUtility += (probabilitySPrime * utilitySPrime);
									}

									if (currentUtility >= maxUtility || maxUtility == Double.MIN_VALUE) {
										maxUtility = currentUtility;
										maxAction = actionPrime;
									}
								}
								/** Calculate the new utility: */
								Double after = reward + (discountFactor * maxUtility);
								/** Recalculate delta by determining the difference, or convergence progression: */
								double convergence = Math.abs(after - expectedValues.get(state));
								if(convergence > delta){
									delta = convergence;
								}
								expectedValues.put(state, after);
							}
						}
					}
				}
			}
		}

		/** Check delta for convergence, termination condition: */
		return delta < (convergenceTolerance * ((1 - discountFactor) / discountFactor));
	}

	public ValueIteratingAgent duplicate()
	{
		ValueIteratingAgent ret = new ValueIteratingAgent();
		ret.setConvergenceTolerance(this.convergenceTolerance);
		ret.setDiscountFactor(this.discountFactor);
		ret.setRewardFunction(this.rewardFunction);
		ret.setTransitionFunction(this.transitionFunction);
		ret.setWorld(this.world);
		ret.expectedValues.putAll(this.expectedValues);
		return ret;
	}
	
	public double getLearningFactor()
	{
		return discountFactor;
	}

	public void setDiscountFactor(double discountFactor)
	{
		this.discountFactor = discountFactor;
	}

	public TransitionFunction getTransitionFunction()
	{
		return transitionFunction;
	}

	public void setTransitionFunction(TransitionFunction transitionFunction)
	{
		this.transitionFunction = transitionFunction;
	}

	public RewardFunction getRewardFunction()
	{
		return rewardFunction;
	}

	public void setRewardFunction(RewardFunction rewardFunction)
	{
		this.rewardFunction = rewardFunction;
	}
	
	public WorldMap getWorld()
	{
		return world;
	}

	public void setWorld(WorldMap world)
	{
		this.world = world;
	}
	
	public double getConvergenceTolerance()
	{
		return convergenceTolerance;
	}

	public void setConvergenceTolerance(double convergenceTolerance)
	{
		this.convergenceTolerance = convergenceTolerance;
	}

	/**
	 * Represents a policy that this agent would produce.
	 */
	public class ValuePolicy implements Policy
	{
		private static final long serialVersionUID = 1L;
		
		private Random random = new Random();

		/**
		 * The action an agent decides to take from a given state 
		 */
		public Action decide(State state)
		{
			// TODO: this function should return an appropriate action based on
			// an exploration policy and the current estimate of expected
			// future reward.
			Double maxUtility = Double.MIN_VALUE;
			Action maxAction = null;
			for (Action currentAction : Action.LEGAL_ACTIONS) {
				/** The transition model function returns a set of all possible actions following state s, along with
				 *  their probability of occurring: */
				Double currentUtility = 0.0;
  				for(Pair<State, Double> sPrime : transitionFunction.transition(state, currentAction)) {
					/** The Utility, U, of each state, multiplied by its probability of occurring:  */
					Double probabilitySPrime = sPrime.getSecond();
					Double utilitySPrime = expectedValues.get(sPrime.getFirst());
					currentUtility += (probabilitySPrime * utilitySPrime);
				}

				if (currentUtility >= maxUtility || maxUtility == Double.MIN_VALUE) {
					maxUtility = currentUtility;
					maxAction = currentAction;
				}
			}

			return maxAction;
		}
	}
}