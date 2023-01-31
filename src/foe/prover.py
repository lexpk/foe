from .logic import (
    Equation,
    Function,
    Variable,
    Substitution,
    mgu,
    Problem,
    Sequent,
    Term,
    get_subterm,
    replace_subterm,
    substitute
)
import heapq as hq


class TermInstance():
    def __init__(
        self,
        sequent_index,
        sequent_side,
        equation_index,
        equation_side,
        subterm_index,
    ):
        self.sequent_index = sequent_index
        self.sequent_side = sequent_side
        self.equation_index = equation_index
        self.equation_side = equation_side
        self.subterm_index = subterm_index

    def __repr__(self):
        return "TermInstance({}, {}, {}, {}, {})".format(
            self.sequent_index,
            self.sequent_side,
            self.equation_index,
            self.equation_side,
            self.subterm_index,
        )

    def __eq__(self, other):
        return (
            self.sequent_index == other.sequent_index
            and self.sequent_side == other.sequent_side
            and self.equation_index == other.equation_index
            and self.equation_side == other.equation_side
            and self.subterm_index == other.subterm_index
        )

    def __lt__(self, other):
        return (
            self.sequent_index,
            self.sequent_side,
            self.equation_index,
            self.equation_side,
            self.subterm_index
        ) < (
            other.sequent_index,
            other.sequent_side,
            other.equation_index,
            other.equation_side,
            other.subterm_index
        )

    def sequent(self, sequentlist: list[Sequent]) -> Sequent:
        """Converts the term instance to the corresponding sequent.

        Parameters
        ----------
        sequentlist : list[Sequent]
            The list of sequents to draw from.

        Returns
        -------
        Sequent
            The sequent corresponding to the term instance.
        """
        return sequentlist[self.sequent_index]

    def equation(self, sequentlist: list[Sequent]) -> Equation:
        """Converts the sequent instance to the corresponding equation.

        Parameters
        ----------
        sequentlist : list[Sequent]
            The list of sequents to draw from.

        Returns
        -------
        Equation
            The equation corresponding to the sequent instance.
        """
        if self.sequent_side == "l":
            eqs = sequentlist[self.sequent_index].antecedent
        elif self.sequent_side == "r":
            eqs = sequentlist[self.sequent_index].succedent
        else:
            raise ValueError(
                "Invalid sequent side: {}. Must be\
                'l' or 'r'.".format(self.sequent_side)
            )
        return eqs[self.equation_index]

    def toplevel(self, sequentlist: list[Sequent]) -> Term:
        """Converts the term instance to the corresponding top level term.

        Parameters
        ----------
        sequentlist : list[Sequent]
            The list of sequents to draw from.

        Returns
        -------
        Term
            The term corresponding to the term instance.
        """
        eq = self.equation(sequentlist)
        if self.equation_side == "l":
            return eq.lhs
        elif self.equation_side == "r":
            return eq.rhs
        else:
            raise ValueError(
                "Invalid equation side: {}. Must be\
                'l' or 'r'.".format(self.equation_side)
            )

    def term(self, sequentlist: list[Sequent]) -> Term:
        """Converts the term instance to the corresponding subterm.

        Parameters
        ----------
        sequentlist : list[Sequent]
            The list of sequents to draw from.

        Returns
        -------
        Term
            The subterm corresponding to the term instance.
        """
        term = self.toplevel(sequentlist)
        return get_subterm(term, self.subterm_index)

    def replace_in_equation(
        self, equation: Equation, t: Term
    ) -> Equation:
        """Replaces the subterm corresponding to the term instance with the
        given term in the corresponding equation.

        Parameters
        ----------
        sequentlist : list[Sequent]
            The list of sequents to draw from.
        t : Term
            The term to replace the subterm with.
        """
        if self.equation_side == "l":
            return Equation(
                replace_subterm(equation.lhs, self.subterm_index, t),
                equation.rhs
            )
        elif self.equation_side == "r":
            return Equation(
                equation.lhs,
                replace_subterm(equation.rhs, self.subterm_index, t)
            )
        else:
            raise ValueError(
                "Invalid equation side: {}. Must be\
                'l' or 'r'.".format(self.equation_side)
            )

    def sequent_without_equation(self, sequentlist: list[Sequent]) -> Sequent:
        """Removes the equation corresponding to the term instance from the
        corresponding sequent.

        Parameters
        ----------
        sequentlist : list[Sequent]
            The list of sequents to draw from.
        """
        seq = self.sequent(sequentlist)
        if self.sequent_side == "l":
            return Sequent(
                seq.antecedent[: self.equation_index] +
                seq.antecedent[self.equation_index + 1:],
                seq.succedent
            )
        elif self.sequent_side == "r":
            return Sequent(
                seq.antecedent,
                seq.succedent[: self.equation_index] +
                seq.succedent[self.equation_index + 1:]
            )
        else:
            raise ValueError(
                "Invalid sequent side: {}. Must be\
                'l' or 'r'.".format(self.sequent_side)
            )


def subtermindices(term: Term) -> list[tuple[int, ...]]:
    """Returns a list of all subterm indices for the given term.

    Parameters
    ----------
    term : Term
        The term to get the subterm indices for.

    Returns
    -------
    list[Tuple]
        The list of subterm indices.
    """
    if isinstance(term, Variable):
        return [tuple()]
    elif isinstance(term, Function):
        indices = [tuple()]
        for i, arg in enumerate(term.arguments):
            for subindex in subtermindices(arg):
                indices.append((i,) + subindex)
        return indices
    else:
        raise ValueError(
            "Invalid term type: {}. Must be\
            Variable or Function.".format(type(term))
        )


def positive_toplevel_terminstances(
    sequentlist: list[Sequent]
) -> list[TermInstance]:
    """Converts a list of sequents to a list of top level term instances.

    Parameters
    ----------
    sequentlist : list[Sequent]
        The list of sequents to convert.

    Returns
    -------
    list[TermInstance]
        The list of top level term instances.
    """
    terminstances = []
    for i, seq in enumerate(sequentlist):
        for j, eq in enumerate(seq.succedent):
            for side in ["l", "r"]:
                terminstances.append(
                    TermInstance(i, "r", j, side, tuple())
                )
    return terminstances


def terminstances(sequentlist: list[Sequent]) -> list[TermInstance]:
    """Converts a list of sequents to a list of term instances.

    Parameters
    ----------
    sequentlist : list[Sequent]
        The list of sequents to convert.

    Returns
    -------
    list[TermInstance]
        The list of term instances.
    """
    terminstances = []
    for i, seq in enumerate(sequentlist):
        for j, eq in enumerate(seq.antecedent):
            for subindex in subtermindices(eq.lhs):
                terminstances.append(
                    TermInstance(i, "l", j, "l", subindex)
                )
            for subindex in subtermindices(eq.rhs):
                terminstances.append(
                    TermInstance(i, "l", j, "r", subindex)
                )
        for j, eq in enumerate(seq.succedent):
            for subindex in subtermindices(eq.lhs):
                terminstances.append(
                    TermInstance(i, "r", j, "l", subindex)
                )
            for subindex in subtermindices(eq.rhs):
                terminstances.append(
                    TermInstance(i, "r", j, "r", subindex)
                )
    return terminstances


class Prover():
    def __init__(self, problem: Problem, evaluator):
        self.problem = problem
        self.sequents = [s.normalize() for s in problem.sequents]
        self.evaluator = evaluator
        self.evaluations = hq.heapify(
            [(self.evaluator(s), s) for s in self.sequents]
        )
        self.superposition_instances: dict[
            Sequent,
            list[(
                TermInstance,
                TermInstance,
                dict[Substitution, Substitution]
            )]
        ] = {s: [] for s in self.sequents}
        self.reverse_superposition_instances: dict[
            Sequent,
            list[(
                TermInstance,
                TermInstance,
                dict[Substitution, Substitution]
            )]
        ] = {s: [] for s in self.sequents}
        for toplevel in positive_toplevel_terminstances(self.sequents):
            for index in terminstances(self.sequents):
                m = mgu(
                    toplevel.toplevel(self.sequents),
                    index.term(self.sequents)
                )
                if m and (
                    toplevel.sequent_index != index.sequent_index
                    or index.subterm_index != ()
                    or index < toplevel
                ):
                    self.superposition_instances[
                        toplevel.sequent(self.sequents)
                    ].append(
                        (toplevel, index, m)
                    )
                    self.reverse_superposition_instances[
                        index.sequent(self.sequents)
                    ].append(
                        (toplevel, index, m)
                    )

    def expand(self, index: int):
        """Expands the sequent at the given index.

        Parameters
        ----------
        index : int
            The index of the sequent to expand.

        Returns
        -------
        list
            The expanded sequents.
        """
        newsequents = set()
        for toplevel, index, m in self.superposition_instances[
            self.sequents[index]
        ]:
            newsequents.add(self.superposition(toplevel, index, m))
        for sequent in newsequents:
            if sequent not in self.sequents:
                self.sequents.append(sequent)
                self.superposition_instances[sequent] = []
                self.reverse_superposition_instances[sequent] = []
                for toplevel in positive_toplevel_terminstances([sequent]):
                    toplevel.sequent_index = len(self.sequents) - 1
                    for index in terminstances(self.sequents):
                        m = mgu(
                            toplevel.toplevel(self.sequents),
                            index.term(self.sequents)
                        )
                        if m and (
                            toplevel.sequent_index != index.sequent_index
                            or index.subterm_index != ()
                            or index < toplevel
                        ):
                            self.superposition_instances[sequent].append(
                                (toplevel, index, m)
                            )
                            self.reverse_superposition_instances[
                                index.sequent(self.sequents)
                            ].append(
                                (toplevel, index, m)
                            )

    def superposition(
        self,
        toplevel: TermInstance,
        index: TermInstance,
        m: tuple[Substitution, Substitution]
    ):
        """Performs superposition on the given sequents.

        Parameters
        ----------
        toplevel : TermInstance
            The top level term instance.
        index : TermInstance
            The term instance to perform superposition with.
        m : tuple[Substitution, Substitution]
            The most general unifier.

        Returns
        -------
        Sequent
            The sequent resulting from superposition.
        """
        newtoplevel = substitute(
            m[0], toplevel.sequent_without_equation(self.sequents)
        )
        newindex = substitute(
            m[1], index.sequent_without_equation(self.sequents)
        )
        if toplevel.sequent_side == "r":
            if index.sequent_side == "l":
                if toplevel.equation_side == "l":
                    return Sequent(
                        newtoplevel.antecedent +
                        newindex.antecedent + (
                            index.replace_in_equation(
                                substitute(
                                    m[1], index.equation(self.sequents)
                                ),
                                substitute(
                                    m[0], toplevel.equation(self.sequents).rhs
                                )
                            ),
                        ),
                        newtoplevel.succedent +
                        newindex.succedent
                    ).normalize()
                elif toplevel.equation_side == "r":
                    return Sequent(
                        newtoplevel.antecedent +
                        newindex.antecedent + (
                            index.replace_in_equation(
                                substitute(
                                    m[1], index.equation(self.sequents)
                                ),
                                substitute(
                                    m[0], toplevel.equation(self.sequents).lhs
                                )
                            ),
                        ),
                        newtoplevel.succedent +
                        newindex.succedent
                    ).normalize()
            elif index.sequent_side == "r":
                if toplevel.equation_side == "l":
                    return Sequent(
                        newtoplevel.antecedent +
                        newindex.antecedent,
                        newtoplevel.succedent +
                        newindex.succedent + (
                            index.replace_in_equation(
                                substitute(
                                    m[1], index.equation(self.sequents)
                                ),
                                substitute(
                                    m[0], toplevel.equation(self.sequents).rhs
                                )
                            ),
                        )
                    ).normalize()
                elif toplevel.equation_side == "r":
                    return Sequent(
                        newtoplevel.antecedent +
                        newindex.antecedent,
                        newtoplevel.succedent +
                        newindex.succedent + (
                            index.replace_in_equation(
                                substitute(
                                    m[1], index.equation(self.sequents)
                                ),
                                substitute(
                                    m[0], toplevel.equation(self.sequents).lhs
                                )
                            ),
                        )
                    ).normalize()
        elif toplevel.sequent_side == "l":
            raise ValueError("Superposition only supports right sequents.")

    def prove(self, maxsteps: int):
        """Attemps a saturation proof of the given problem.

        Parameters
        ----------
        maxsteps : int
            The maximum number of steps to take.

        Returns
        -------
        bool
            True if the sequents are provable, False otherwise.
        """
        pass
