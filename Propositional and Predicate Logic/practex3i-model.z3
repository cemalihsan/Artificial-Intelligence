(set-option :produce-models true)

; Defining the predicates:
(declare-fun P (Int Int) Bool)
(declare-fun Q (Int Int) Bool)
(declare-fun R (Int Int) Bool)

;; TODO: model exercise 3(i) here
(assert (forall ((x Int)) (exists ((y Int)) (P x y))))
(assert (forall ((x Int)) (exists ((y Int)) (not (R x y)))))
(assert (forall ((x Int)) (exists ((y Int)) (=> (or (P x y) (not (R x y))) (forall ((z Int)) (or (Q x z) (and (not (P y z)) (R y z))))))))
(assert (forall ((x Int)) (exists ((y Int)) (Q x y))))
(check-sat)
