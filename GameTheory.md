# 1. Essential elements of a game in game theory. 
Game theory is a branch of mathematics that studies strategic interactions between decision-makers, known as “players.” It analyzes situations where the outcome for each participant depends on the choices made by all involved. This field provides tools to model and predict the behavior of players in competitive and cooperative scenarios
In game theory, the essential elements of a game are: 
1. Players: Decision-makers in the game. 
2. Strategies: The complete plan of action a player can take. 
3. Payoffs: Rewards or outcomes received based on chosen strategies. 
4. Information: What players know at various stages of the game. 
5. Actions: Specific choices available to players. 
6. Outcomes: The results of the players' combined actions. 
7. Rules of the Game: The structure and framework governing the game. 
8. Equilibrium: A stable state where no player benefits from changing their strategy.

# 2. Non-Cooperative Game Theory
- Definition: 
Focuses on how rational players make decisions independently, often in competitive settings where binding agreements are not possible.
- Key Features:
    - Individual Strategies: 
    Each player chooses their strategy to maximize their own payoff, considering the strategies of others.
    - Nash Equilibrium: 
    A common solution concept where no player can benefit by changing their strategy unilaterally.
- Examples: 
Prisoner’s Dilemma, Cournot Competition.
Applications: Used in economics, political science, and evolutionary biology to model competitive behaviors and conflicts.

# 3. Cooperative Game Theory
- Definition: 
Focuses on how players can form coalitions and make binding agreements to achieve a common goal, often in collaborative settings.
- Key Features:
    - Coalitions: 
    Players form groups (coalitions) to improve their collective outcomes.
- Examples: 
Bargaining games, coalition formation in politics.
Applications: Used in economics, business, and social sciences to model collaborative behaviors and negotiations

# 4. Key diffrences
![alt text](image.png)

# 5. The Prisoner’s Dilemma: 
is a fundamental problem in game theory that demonstrates why two rational individuals might not cooperate, even if it seems that cooperation would be 
in their best interest. 

> Scenario 

Imagine two criminals, Alice and Bob, who are arrested and interrogated separately. The 
police lack sufficient evidence to convict either of a major crime, but they can convict them 
of a lesser offense. The criminals are offered a deal: 
1. If Alice betrays Bob (defects) and Bob remains silent (cooperates), Alice goes free, and Bob gets 10 years in prison. 
2. If Bob betrays Alice and Alice remains silent, Bob goes free, and Alice gets 10 years. 
3. If both betray each other, they both get 5 years in prison. 
4. If both remain silent, they each get 1 year for the lesser offense. 

# 6. Game theory Applications:
Below are two notable applications:

- **Auction Design and Bidding Strategies**: 

    Game theory plays a crucial role in designing and analyzing auctions, particularly in industries where companies bid for resources, licenses, or contracts. For instance, governments use auctions to allocate spectrum licenses to telecommunications companies, and energy markets use auctions to manage electricity supply.

    Game Theory Application: In auctions, companies need to decide how much to bid for a resource, considering both their valuation of the resource and the potential bids of competitors. Game theory models, such as the first-price sealed-bid auction and second-price (Vickrey) auction, help predict bidding strategies.

    Outcome: Game theory provides insights into how bidders strategize, the likely outcomes, and how the auction format can affect overall efficiency and revenue. For instance, in spectrum auctions, this analysis helps governments design auctions that maximize revenue while ensuring fair competition and efficient resource allocation.

- **Oligopolistic Competition and Pricing Strategies** :

    In highly competitive markets, companies often face decisions about pricing their products or services. Game theory provides a framework to analyze how companies interact when making these decisions, considering the potential reactions of competitors. This is particularly evident in industries where only a few firms dominate, like airlines, telecommunications, or retail.

- **International Relations and Conflict Resolution**:

    In international relations, countries often face decisions about forming alliances, engaging in trade, or escalating conflicts. Game theory is used to analyze the strategic interactions between nations, especially when decisions involve high stakes, like nuclear deterrence or trade tariffs.

# 7. Rational choice theory: 
Rational Choice Theory (RCT) is a framework used in economics, sociology, and political science to understand and model human behavior. It assumes that individuals make decisions by weighing the costs and benefits to maximize their personal advantage. This theory assumes that people are rational actors who aim to achieve the most favorable outcome based on their preferences and available information. 

Example: Imagine a student deciding whether to study for an exam or go to a party. According to RCT, the student will evaluate the potential outcomes of both choices. Studying might increase the likelihood of a good grade (which the student values), while going to the party might provide immediate enjoyment but could harm academic performance. If the student values academic success more and believes studying will provide long-term benefits, they will likely choose to study. Viceversa ....


# 8. Strategy: 
is a set of rules or actions a player commits to, which dictates how they will respond to various situations or actions taken by other players.
> Types: 

- Pure Strategy: A specific, predetermined action a player will take in every possible situation (e.g., always choosing to cooperate in the Prisoner’s 
Dilemma). 

-  Mixed Strategy: A probabilistic approach where a player chooses among different actions according to a set probability distribution (e.g., randomly choosing between several actions to keep opponents uncertain). 

> Example: Rock-Paper-Scissors: 
- Pure Strategy: Always choose Rock. 
- Mixed Strategy: Choose Rock, Paper, or Scissors with equal probability (1/3 each). 


# 9. A pure strategy Nash Equilibrium: 
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

# 10. Classification of Game Theory: 
> Game theory can be classified based on various criteria: 

**1. By Number of Players**: 
- Two-Player Games: Games involving exactly two players (e.g., Prisoner’s Dilemma). 
- N-Player Games: Games involving more than two players (e.g., Public Goods Games). 

**2. By Type of Strategies**: 
- Pure Strategy Games: Each player chooses a single strategy with certainty 
(e.g., Rock-Paper-Scissors). 

- Mixed Strategy Games: Players choose among strategies with certain probabilities (e.g., randomized strategies in games like poker). 

- Dominant Strategy: A dominant strategy is one that always provides a better outcome for a player, regardless of what the other players do. It is the best choice in every possible scenario. 

    - Example: Prisoner’s Dilemma 
        Defect (betray) is a dominant strategy for both players because it leads to a better or equal outcome compared to cooperating, regardless of the other player’s action. 

**3. By Information Availability**: 
- **Complete / perfect Information**: games where all players have 
complete knowledge of the game's structure and all previous actions taken by other players. There are no hidden information or unknown moves; each player isfully informed about the game's state at all times. 

- Incomplete Information: Players have limited knowledge about other players' payoffs or strategies (e.g., auctions with unknown valuations). 

**4. By Timing of Moves**: 
- Simultaneous-Move Games: Players make decisions without knowing the other players’ choices (e.g., The Prisoner’s Dilemma). 

- Sequential-Move Games: Players make decisions one after another, with later players having knowledge of earlier moves (e.g., Chess). 

**5. By Type of Payoff Structure**: 
- Zero-Sum Games: One player’s gain is exactly balanced by the losses of others (e.g., Poker). 
- Non-Zero-Sum Games: The total gains and losses are not necessarily balanced, allowing for mutually beneficial outcomes (e.g., Trade Negotiations). 


# 11. Saddle point:
In game theory, a saddle point (or minimax point) is a crucial concept in zero-sum games. It represents a situation where the chosen strategies of both players intersect, leading to an optimal outcome for both, given the constraints of the game. 

A saddle point in a payoff matrix is an entry that is:
The smallest value in its row (minimizing the maximum loss for the row player).
The largest value in its column (maximizing the minimum gain for the column player).


# 12. Oligopoly: 
is a market structure characterized by a small number of firms that have significant market power. Each firm's decisions affect the others, leading to strategic 
interactions. Oligopoly models analyze how firms in such markets make decisions regarding pricing, output, and other strategic variables. 
> Here are some key oligopoly models: 

1. Cournot Model: 
The Cournot Model focuses on quantity competition. Firms decide how much to produce, and the market price is determined by the total quantity produced. 

    > Key Features: 
    - Firms produce a homogeneous product. 
    - Each firm chooses its output level to maximize profit, assuming the output levels of other firms are fixed. 
    - Firms have complete information about market demand and production costs. 
    > The equilibrium is reached when no firm can increase its profit by changing its output, given the output of the other firms.

2. Bertrand Model: 
The Bertrand Model focuses on price competition. Firms compete by setting prices rather than quantities. The model assumes that firms produce identical products. 

    > Key Features: 
    
