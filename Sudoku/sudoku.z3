(set-option :produce-models true)
(declare-datatypes () ((Col x0 x1 x2 x3 x4 x5 x6 x7 x8)))
(declare-datatypes () ((Row y0 y1 y2 y3 y4 y5 y6 y7 y8)))
(declare-fun Board (Col Row) Int)


; Fill in the missing constraints into the designated sections


;;;;;; --- BEGIN-CONSTRAINTS-1 --- 

; ** Sudoku constraints **
; TODO constraints for 1.

(assert(distinct(Board x0 y0) (Board x1 y0) (Board x2 y0) (Board x3 y0) (Board x4 y0) (Board x5 y0) (Board x6 y0) (Board x7 y0) (Board x8 y0)))

(assert(distinct(Board x0 y1) (Board x1 y1) (Board x2 y1) (Board x3 y1) (Board x4 y1) (Board x5 y1) (Board x6 y1) (Board x7 y1) (Board x8 y1)))

(assert(distinct(Board x0 y2) (Board x1 y2) (Board x2 y2) (Board x3 y2) (Board x4 y2) (Board x5 y2) (Board x6 y2) (Board x7 y2) (Board x8 y2)))

(assert(distinct(Board x0 y3) (Board x1 y3) (Board x2 y3) (Board x3 y3) (Board x4 y3) (Board x5 y3) (Board x6 y3) (Board x7 y3) (Board x8 y3)))

(assert(distinct(Board x0 y4) (Board x1 y4) (Board x2 y4) (Board x3 y4) (Board x4 y4) (Board x5 y4) (Board x6 y4) (Board x7 y4) (Board x8 y4)))

(assert(distinct(Board x0 y5) (Board x1 y5) (Board x2 y5) (Board x3 y5) (Board x4 y5) (Board x5 y5) (Board x6 y5) (Board x7 y5) (Board x8 y5)))

(assert(distinct(Board x0 y6) (Board x1 y6) (Board x2 y6) (Board x3 y6) (Board x4 y6) (Board x5 y6) (Board x6 y6) (Board x7 y6) (Board x8 y6)))

(assert(distinct(Board x0 y7) (Board x1 y7) (Board x2 y7) (Board x3 y7) (Board x4 y7) (Board x5 y7) (Board x6 y7) (Board x7 y7) (Board x8 y7)))

(assert(distinct(Board x0 y8) (Board x1 y8) (Board x2 y8) (Board x3 y8) (Board x4 y8) (Board x5 y8) (Board x6 y8) (Board x7 y8) (Board x8 y8)))

(assert(=(Board x0 y0)6))
(assert(=(Board x0 y2)2))
(assert(=(Board x2 y1)9))
(assert(=(Board x2 y2)4))

;;;;;; --- END-CONSTRAINTS-1 ---



;;;;;; --- BEGIN-CONSTRAINTS-2 --- 

; ** Constraints for top middle square **
; TODO constraints for 2.

(assert(or(=(+ 1(Board x3 y0))(Board x3 y1))(=(+ 1(Board x3 y0))(Board x4 y0))))
(assert(or(=(+ 1(Board x5 y0))(Board x4 y0))(=(+ 1(Board x5 y0))(Board x4 y1))))
(assert(or(=(+ 1(Board x3 y2))(Board x3 y1))(=(+ 1(Board x3 y2))(Board x4 y2))))
(assert(or(=(+ 1(Board x5 y2))(Board x4 y1))(=(+ 1(Board x5 y2))(Board x4 y2))))
;;;;;; --- END-CONSTRAINTS-2 ---



;;;;;; --- BEGIN-CONSTRAINTS-3 --- 

; ** Constraints for top right square **
; TODO constraints for 3.
(assert(=(+(Board x8 y1)(Board x7 y0)(Board x6 y1)(Board x7 y2))(* 2(Board
x7 y1))))

;;;;;; --- END-CONSTRAINTS-3 ---


;;;;;; --- BEGIN-CONSTRAINTS-4 --- 

; ** Constraints for middle left square **
; TODO constraints for 4.
(assert(=(+(Board x1 y3)(Board x2 y3))13))
(assert(=(Board x0 y5)(* (Board x0 y4)2)))
(assert(=(-(Board x2 y5)(Board x2 y4))4))

;;;;;; --- END-CONSTRAINTS-4 ---



;;;;;; --- BEGIN-CONSTRAINTS-5 --- 

; ** Constraints for center square **
; TODO constraints for 5.
(assert(>(Board x3 y3)(Board x3 y4)))
(assert(>(Board x3 y4)(Board x3 y5)))
(assert(>(Board x4 y3)(Board x4 y4)))
(assert(>(Board x4 y4)(Board x4 y5)))
(assert(>(Board x5 y3)(Board x5 y4)))
(assert(>(Board x5 y4)(Board x5 y5)))
;;;;;; --- END-CONSTRAINTS-5 ---



;;;;;; --- BEGIN-CONSTRAINTS-6 --- 

; ** Constraints for middle right square **
; TODO constraints for 6.

(assert (or 
(and (< (Board x7 y3) 5) (>= (Board x6 y4) 5) (>= (Board x7 y5) 5) (>= (Board x8 y4) 5)) 
(and (< (Board x6 y4) 5) (>= (Board x7 y3) 5) (>= (Board x7 y5) 5) (>= (Board x8 y4) 5)) 
(and (< (Board x7 y5) 5) (>= (Board x6 y4) 5) (>= (Board x7 y3) 5) (>= (Board x8 y4) 5)) 
(and (< (Board x8 y4) 5) (>= (Board x6 y4) 5) (>= (Board x7 y5) 5) (>= (Board x7 y3) 5))))

;;;;;; --- END-CONSTRAINTS-6 ---



;;;;;; --- BEGIN-CONSTRAINTS-7 --- 

; ** Constraints for bottom left square **
; TODO constraints for 7.
(assert(=(+(Board x0 y6)(Board x1 y6)(Board x2 y6))(+(Board x0 y7)(Board x1 y7)(Board x2 y7))))
(assert(=(+(Board x0 y6)(Board x1 y6)(Board x2 y6))(+(Board x0 y8)(Board x1 y8)(Board x2 y8))))
(assert(=(+(Board x0 y8)(Board x1 y8)(Board x2 y8))(+(Board x0 y7)(Board x1 y7)(Board x2 y7))))

;;;;;; --- END-CONSTRAINTS-7 ---



;;;;;; --- BEGIN-CONSTRAINTS-8 --- 

; ** Constraints for bottom middle square **
; TODO constraints for 8.

;(a) If one of the multiplicants is odd, then the product must be odd.
(assert(or (and (distinct 2 4 6 8 10 12 14 16 18 20 22 24 (+(Board x3 y6)(Board x3 y7)(Board x3 y8)))
                (distinct 1 3 5 7 9 11 13 15 17 19 21 23  (+(Board x5 y6)(Board x5 y7)(Board x5 y8))))
           (and (distinct 1 3 5 7 9 11 13 15 17 19 21 23  (+(Board x3 y6)(Board x3 y7)(Board x3 y8)))
                (distinct 2 4 6 8 10 12 14 16 18 20 22 24 (+(Board x5 y6)(Board x5 y7)(Board x5 y8))))))
;(b)
(assert(= (+(Board x4 y7)(Board x4 y8)) 3))

;;;;;; --- END-CONSTRAINTS-8 ---



;;;;;; --- BEGIN-CONSTRAINTS-9 --- 

; ** Constraints for bottom right square **
; TODO constraints for 9.

(assert(or
  (and ; all green squares are odd & orange cell is even
    (distinct 2 4 6 8 (Board x6 y6))
    (distinct 2 4 6 8 (Board x6 y8))
    (distinct 2 4 6 8 (Board x8 y6))
    (distinct 2 4 6 8 (Board x8 y8))
    (distinct 1 3 5 7 9 (Board x7 y7))
  )
  (and ; all green squares are even & orange cell is odd
    (distinct 1 3 5 7 9 (Board x6 y6))
    (distinct 1 3 5 7 9 (Board x6 y8))
    (distinct 1 3 5 7 9 (Board x8 y6))
    (distinct 1 3 5 7 9 (Board x8 y8))
    (distinct 2 4 6 8   (Board x7 y7))
  )
))

;;;;;; --- END-CONSTRAINTS-9 ---





;;;;;; --- BEGIN-CONSTRAINTS-DOMAIN ---

; Each cell may be assigned only to integer values between 1 and 9:

(assert (and (< (Board x0 y0) 10) (> (Board x0 y0) 0)))
(assert (and (< (Board x0 y1) 10) (> (Board x0 y1) 0)))
(assert (and (< (Board x0 y2) 10) (> (Board x0 y2) 0)))
(assert (and (< (Board x0 y3) 10) (> (Board x0 y3) 0)))
(assert (and (< (Board x0 y4) 10) (> (Board x0 y4) 0)))
(assert (and (< (Board x0 y5) 10) (> (Board x0 y5) 0)))
(assert (and (< (Board x0 y6) 10) (> (Board x0 y6) 0)))
(assert (and (< (Board x0 y7) 10) (> (Board x0 y7) 0)))
(assert (and (< (Board x0 y8) 10) (> (Board x0 y8) 0)))
(assert (and (< (Board x1 y0) 10) (> (Board x1 y0) 0)))
(assert (and (< (Board x1 y1) 10) (> (Board x1 y1) 0)))
(assert (and (< (Board x1 y2) 10) (> (Board x1 y2) 0)))
(assert (and (< (Board x1 y3) 10) (> (Board x1 y3) 0)))
(assert (and (< (Board x1 y4) 10) (> (Board x1 y4) 0)))
(assert (and (< (Board x1 y5) 10) (> (Board x1 y5) 0)))
(assert (and (< (Board x1 y6) 10) (> (Board x1 y6) 0)))
(assert (and (< (Board x1 y7) 10) (> (Board x1 y7) 0)))
(assert (and (< (Board x1 y8) 10) (> (Board x1 y8) 0)))
(assert (and (< (Board x2 y0) 10) (> (Board x2 y0) 0)))
(assert (and (< (Board x2 y1) 10) (> (Board x2 y1) 0)))
(assert (and (< (Board x2 y2) 10) (> (Board x2 y2) 0)))
(assert (and (< (Board x2 y3) 10) (> (Board x2 y3) 0)))
(assert (and (< (Board x2 y4) 10) (> (Board x2 y4) 0)))
(assert (and (< (Board x2 y5) 10) (> (Board x2 y5) 0)))
(assert (and (< (Board x2 y6) 10) (> (Board x2 y6) 0)))
(assert (and (< (Board x2 y7) 10) (> (Board x2 y7) 0)))
(assert (and (< (Board x2 y8) 10) (> (Board x2 y8) 0)))
(assert (and (< (Board x3 y0) 10) (> (Board x3 y0) 0)))
(assert (and (< (Board x3 y1) 10) (> (Board x3 y1) 0)))
(assert (and (< (Board x3 y2) 10) (> (Board x3 y2) 0)))
(assert (and (< (Board x3 y3) 10) (> (Board x3 y3) 0)))
(assert (and (< (Board x3 y4) 10) (> (Board x3 y4) 0)))
(assert (and (< (Board x3 y5) 10) (> (Board x3 y5) 0)))
(assert (and (< (Board x3 y6) 10) (> (Board x3 y6) 0)))
(assert (and (< (Board x3 y7) 10) (> (Board x3 y7) 0)))
(assert (and (< (Board x3 y8) 10) (> (Board x3 y8) 0)))
(assert (and (< (Board x4 y0) 10) (> (Board x4 y0) 0)))
(assert (and (< (Board x4 y1) 10) (> (Board x4 y1) 0)))
(assert (and (< (Board x4 y2) 10) (> (Board x4 y2) 0)))
(assert (and (< (Board x4 y3) 10) (> (Board x4 y3) 0)))
(assert (and (< (Board x4 y4) 10) (> (Board x4 y4) 0)))
(assert (and (< (Board x4 y5) 10) (> (Board x4 y5) 0)))
(assert (and (< (Board x4 y6) 10) (> (Board x4 y6) 0)))
(assert (and (< (Board x4 y7) 10) (> (Board x4 y7) 0)))
(assert (and (< (Board x4 y8) 10) (> (Board x4 y8) 0)))
(assert (and (< (Board x5 y0) 10) (> (Board x5 y0) 0)))
(assert (and (< (Board x5 y1) 10) (> (Board x5 y1) 0)))
(assert (and (< (Board x5 y2) 10) (> (Board x5 y2) 0)))
(assert (and (< (Board x5 y3) 10) (> (Board x5 y3) 0)))
(assert (and (< (Board x5 y4) 10) (> (Board x5 y4) 0)))
(assert (and (< (Board x5 y5) 10) (> (Board x5 y5) 0)))
(assert (and (< (Board x5 y6) 10) (> (Board x5 y6) 0)))
(assert (and (< (Board x5 y7) 10) (> (Board x5 y7) 0)))
(assert (and (< (Board x5 y8) 10) (> (Board x5 y8) 0)))
(assert (and (< (Board x6 y0) 10) (> (Board x6 y0) 0)))
(assert (and (< (Board x6 y1) 10) (> (Board x6 y1) 0)))
(assert (and (< (Board x6 y2) 10) (> (Board x6 y2) 0)))
(assert (and (< (Board x6 y3) 10) (> (Board x6 y3) 0)))
(assert (and (< (Board x6 y4) 10) (> (Board x6 y4) 0)))
(assert (and (< (Board x6 y5) 10) (> (Board x6 y5) 0)))
(assert (and (< (Board x6 y6) 10) (> (Board x6 y6) 0)))
(assert (and (< (Board x6 y7) 10) (> (Board x6 y7) 0)))
(assert (and (< (Board x6 y8) 10) (> (Board x6 y8) 0)))
(assert (and (< (Board x7 y0) 10) (> (Board x7 y0) 0)))
(assert (and (< (Board x7 y1) 10) (> (Board x7 y1) 0)))
(assert (and (< (Board x7 y2) 10) (> (Board x7 y2) 0)))
(assert (and (< (Board x7 y3) 10) (> (Board x7 y3) 0)))
(assert (and (< (Board x7 y4) 10) (> (Board x7 y4) 0)))
(assert (and (< (Board x7 y5) 10) (> (Board x7 y5) 0)))
(assert (and (< (Board x7 y6) 10) (> (Board x7 y6) 0)))
(assert (and (< (Board x7 y7) 10) (> (Board x7 y7) 0)))
(assert (and (< (Board x7 y8) 10) (> (Board x7 y8) 0)))
(assert (and (< (Board x8 y0) 10) (> (Board x8 y0) 0)))
(assert (and (< (Board x8 y1) 10) (> (Board x8 y1) 0)))
(assert (and (< (Board x8 y2) 10) (> (Board x8 y2) 0)))
(assert (and (< (Board x8 y3) 10) (> (Board x8 y3) 0)))
(assert (and (< (Board x8 y4) 10) (> (Board x8 y4) 0)))
(assert (and (< (Board x8 y5) 10) (> (Board x8 y5) 0)))
(assert (and (< (Board x8 y6) 10) (> (Board x8 y6) 0)))
(assert (and (< (Board x8 y7) 10) (> (Board x8 y7) 0)))
(assert (and (< (Board x8 y8) 10) (> (Board x8 y8) 0)))

;;;;;; --- END-CONSTRAINTS-DOMAIN ---



; Compute solution

(check-sat)



; Print solution, line by line

(get-value ((Board x0 y0)))
(get-value ((Board x1 y0)))
(get-value ((Board x2 y0)))
(get-value ((Board x3 y0)))
(get-value ((Board x4 y0)))
(get-value ((Board x5 y0)))
(get-value ((Board x6 y0)))
(get-value ((Board x7 y0)))
(get-value ((Board x8 y0)))
(get-value ((Board x0 y1)))
(get-value ((Board x1 y1)))
(get-value ((Board x2 y1)))
(get-value ((Board x3 y1)))
(get-value ((Board x4 y1)))
(get-value ((Board x5 y1)))
(get-value ((Board x6 y1)))
(get-value ((Board x7 y1)))
(get-value ((Board x8 y1)))
(get-value ((Board x0 y2)))
(get-value ((Board x1 y2)))
(get-value ((Board x2 y2)))
(get-value ((Board x3 y2)))
(get-value ((Board x4 y2)))
(get-value ((Board x5 y2)))
(get-value ((Board x6 y2)))
(get-value ((Board x7 y2)))
(get-value ((Board x8 y2)))
(get-value ((Board x0 y3)))
(get-value ((Board x1 y3)))
(get-value ((Board x2 y3)))
(get-value ((Board x3 y3)))
(get-value ((Board x4 y3)))
(get-value ((Board x5 y3)))
(get-value ((Board x6 y3)))
(get-value ((Board x7 y3)))
(get-value ((Board x8 y3)))
(get-value ((Board x0 y4)))
(get-value ((Board x1 y4)))
(get-value ((Board x2 y4)))
(get-value ((Board x3 y4)))
(get-value ((Board x4 y4)))
(get-value ((Board x5 y4)))
(get-value ((Board x6 y4)))
(get-value ((Board x7 y4)))
(get-value ((Board x8 y4)))
(get-value ((Board x0 y5)))
(get-value ((Board x1 y5)))
(get-value ((Board x2 y5)))
(get-value ((Board x3 y5)))
(get-value ((Board x4 y5)))
(get-value ((Board x5 y5)))
(get-value ((Board x6 y5)))
(get-value ((Board x7 y5)))
(get-value ((Board x8 y5)))
(get-value ((Board x0 y6)))
(get-value ((Board x1 y6)))
(get-value ((Board x2 y6)))
(get-value ((Board x3 y6)))
(get-value ((Board x4 y6)))
(get-value ((Board x5 y6)))
(get-value ((Board x6 y6)))
(get-value ((Board x7 y6)))
(get-value ((Board x8 y6)))
(get-value ((Board x0 y7)))
(get-value ((Board x1 y7)))
(get-value ((Board x2 y7)))
(get-value ((Board x3 y7)))
(get-value ((Board x4 y7)))
(get-value ((Board x5 y7)))
(get-value ((Board x6 y7)))
(get-value ((Board x7 y7)))
(get-value ((Board x8 y7)))
(get-value ((Board x0 y8)))
(get-value ((Board x1 y8)))
(get-value ((Board x2 y8)))
(get-value ((Board x3 y8)))
(get-value ((Board x4 y8)))
(get-value ((Board x5 y8)))
(get-value ((Board x6 y8)))
(get-value ((Board x7 y8)))
(get-value ((Board x8 y8)))
