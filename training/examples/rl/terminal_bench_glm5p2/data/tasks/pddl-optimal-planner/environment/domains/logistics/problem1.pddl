(define (problem logistics-cross-city)
  (:domain logistics)
  (:objects
    apn1 - airplane
    apt1 apt2 - airport
    pos1 pos2 - location
    cit1 cit2 - city
    tru1 tru2 - truck
    pkg1 - package)
  (:init (at apn1 apt1)
         (at tru1 pos1) (at tru2 pos2)
         (at pkg1 pos1)
         (in-city pos1 cit1) (in-city apt1 cit1)
         (in-city pos2 cit2) (in-city apt2 cit2))
  (:goal (at pkg1 pos2))
)
