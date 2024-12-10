# `UNIT - 2`

# 1. Strategic games: 
are a central concept in Game Theory, where multiple players make decisions simultaneously, taking into account the potential choices of others. The outcomes depend on the strategies of all players, highlighting the interplay between competition and cooperation.


# 2. The Prisoner’s Dilemma: 
is a fundamental problem in game theory that demonstrates why two rational individuals might not cooperate, even if it seems that cooperation would be 
in their best interest. 

## Scenario 

Imagine two criminals, Alice and Bob, who are arrested and interrogated separately. The 
police lack sufficient evidence to convict either of a major crime, but they can convict them 
of a lesser offense. The criminals are offered a deal: 
1. If Alice betrays Bob (defects) and Bob remains silent (cooperates), Alice goes free, and Bob gets 10 years in prison. 
2. If Bob betrays Alice and Alice remains silent, Bob goes free, and Alice gets 10 years. 
3. If both betray each other, they both get 5 years in prison. 
4. If both remain silent, they each get 1 year for the lesser offense. 


# 3. The Matching Pennies game: 
is a classic example in game theory that illustrates conflict, mixed strategies, and the concept of a zero-sum game. Here's a detailed breakdown:

## Setup of the Game

- Players: Two players (let's call them Player A and Player B).
- Strategies:
Each player chooses one of two actions: Heads (H) or Tails (T).
- Payoffs:
    - If the two pennies match (both are Heads or both are Tails), Player A wins and takes $1 from Player B.
    - If the pennies do not match (one is Heads, the other is Tails), Player B wins and takes $1 from Player A.
- Payoff Matrix
The payoffs for Player A (Player B's payoffs are the negatives of these, as it's a zero-sum game) are as follows:

                                Player B: Heads (H)      |     Player B: Tails (T)
        Player A: Heads (H)	         +1 (A wins)	     |          -1 (B wins)
        Player A: Tails (T)	         -1 (B wins)	     |          +1 (A wins)


## Characteristics of the Game
- Zero-Sum Game: One player's gain is exactly equal to the other's loss.
- No Pure Strategy Nash Equilibrium
- Mixed Strategy Nash Equilibrium:
    - The optimal strategy for both players is to randomize their choices, selecting Heads or Tails with equal probability (50% each).
    - By doing so, neither player can predict or exploit the other’s choice, ensuring a balance.

# 4. A pure strategy Nash Equilibrium: 
is a situation in a game where each player chooses a single, specific strategy, and no player can benefit by unilaterally changing their strategy, given that 
other players stick to their chosen strategies. In other words, it's a stable state where each 
player's strategy is the best response to the strategies of the others. 

>Example: The Prisoner’s Dilemma
Consider the classic Prisoner’s Dilemma:

- Players: Two criminals, Alice and Bob.
Strategies: Each can either Confess © or Stay Silent (S).

- Payoffs:
If both confess, they each get 5 years in prison.
If one confesses and the other stays silent, the confessor goes free, and the silent one gets 10 years. If both stay silent, they each get 1 year in prison.

> Finding the Nash Equilibrium

- Alice’s Best Response:

    - If Bob confesses ©, Alice’s best response is to confess © because -5 (confess) is better than -10 (stay silent).
    - If Bob stays silent (S), Alice’s best response is to confess © because 0 (confess) is better than -1 (stay silent).

- Bob’s Best Response:
    - If Alice confesses ©, Bob’s best response is to confess © because -5 (confess) is better than -10 (stay silent).
    - If Alice stays silent (S), Bob’s best response is to confess © because 0 (confess) is better than -1 (stay silent).

Since both Alice and Bob’s best responses are to confess regardless of the other’s choice, (C, C) is a pure strategy Nash Equilibrium. Neither Alice nor Bob can improve their situation by unilaterally changing their strategy.


# `UNIT - 3`


# 1. Difference between perfect and imperfect info games

## Games with Perfect Information
all players are fully aware of the history of the game, including all actions taken by all players up to the current point. There is no hidden information. Examples:
- Chess
- Tik-tak-toe
- Checkers

## Games with Imperfect Information: 
are games in game theory where at least one player does not have complete knowledge about certain aspects of the game at some point during play. Examples: 
- Poker
- Rock Papper Scisssor
- Economic Markets
- Auctions
- Battleship


# 2. Bayesian games: 
are a type of game in game theory where players have incomplete information about the other players. This means that each player has private information that others do not know, such as their preferences, payoffs, or strategies. The concept was introduced by John C. Harsanyi, who won the Nobel Prize in Economics for his contributions to game theory. Key Elements of Bayesian Games:
- Players (N): 
The set of players involved in the game.
- Actions (A): 
The set of actions available to each player.
- Types (T): 
The set of possible types for each player, representing their private information.
- Payoff Functions (u): 
The payoffs each player receives, which depend on their type and the actions taken.
- Prior (p): 
A probability distribution over the possible types, representing the players’ beliefs about each other’s types.

## Applications of Bayesian Games
- Auctions and Bidding:
Modeling competitive bidding in real-world auctions like online ad placements or spectrum auctions.
- Negotiations:
When parties have private valuations or costs.
- Contract Design:
Employers designing contracts without full knowledge of employees’ preferences or abilities.
- Political Science:
Modeling elections where candidates have private policy preferences
    
## Example of a Bayesian Game: 
Let’s consider a simple sealed-bid auction in which two players (Player 1 and Player 2) are bidding for an object. Each player has a private valuation of the object, which is unknown to the other player. The players bid without knowing each other's valuation and the highest bid wins the auction. If a player wins, they get the object and pay the amount they bid, and if they lose, they get nothing.


