# Σχεδιασμός Ενεργειών, Χρονοπρογραμματισμός και ικανοποίηση περιορισμών

Ονοματεπώνυμο: Μερτζεμεκιανός Μάρκος
ΑΕΜ: 212

## Ανάλυση προβλήματος

Στο παρακάτω πρόβλημα αναγνωρίζονται οι παρακάτω οντότητες
- Το φορτηγό
- Η προβλήτα( ή πλατφόρμα )
- Το container

Ενώ τα κατηγορήματα που επιλέχθηκαν είναι τα εξής
- on 
- adj
- free
- has_container_below

Αυτά τα κατηγορήματα αρκούν για να περιγράψουν τις σχέσεις μεταξύ των οντοτήτων. Κάποια κατηγορήματα επίσης χρησιμοποιήθηκαν για να χαρακτηρίσου τις καταστάσεις των οντοτήτων.

Ένα φορτηγό είναι ελεύθερο όταν δεν έχει container πάνω του. Μία πλατφόρμα είναι ελεύθερη όταν δεν έχει φορτηγό πάνω της. Ενώ ένα container είναι free όταν μπορεί να μετακινηθεί από το φορτηγό.

Το on περιγράφει
- ένα φορτηγό είναι πάνω σε μία πλατφόρμα
- ένα container είναι πάνω σε ενα φορτηγό
- ένα container είναι πάνω σε μία πλατφόρμα
- ένα container είναι πάνω σε ένα container

Το adj περιγράφει τη σχέση ότι δύο πλατφόρμες είναι γειτονικές.

Το has_container_below περιγράφει τη σχέση ότι ένα container έχει ένα container από κάτω του.

## Σχεδιασμός ενεργειών

Οι ενεργειές που επιλέχθηκαν είναι οι εξής:

### move_truck (?t ?source ?target)

To move_truck είναι μία ενέργεια που μετακινεί ένα φορτηγό από μία πλατφόρμα σε μία άλλη πλατφόρμα.
Βασική προϋπόθεση είναι η target πλατφόρμα να είναι ελεύθερη. Το αποτέλεσμα είναι το φορτηγό να μετακινηθεί στη target πλατφόρμα, όπου δεν θα είναι και ελεύθερη πλέον μετά την μετακίνηση, και η source να ελευθερωθεί.

| Προϋποθέσεις | Αποτελέσματα |
|--------------|---------------|
| truck(?t), platform(?source), platform(?target) | <span style="background-color:#90EE90;color:black">on(?t, ?target)</span> |
| adj(?source, ?target) ∨ adj(?target, ?source) | <span style="background-color:#90EE90;color:black">free(?source)</span> |
| free(?target), on(?t, ?source) | <span style="background-color:#f8d7da;color:black">not free(?target)</span>, <span style="background-color:#f8d7da;color:black">not on(?t, ?source)</span> |

### exchange_pos_truck (?t1 ?t2 ?p1 ?p2)

Το exchange_pos_truck είναι μία ενέργεια που ανταλλάσσει τη θέση δύο φορτηγών μεταξύ δύο πλατφόρμων.
Βασική προϋπόθεση είναι οι δύο πλατφόρμες να είναι γειτονικές και τα δύο φορτηγά να βρίσκονται στις αντίστοιχες πλατφόρμες.
Το αποτέλεσμα είναι τα φορτηγά να ανταλλάσσουν τη θέση τους και να βρίσκονται στις αντίστοιχες πλατφόρμες.
Αυτή η ενέργεια δημιουργήθηκε διότι ενώ στις πλατφόρμε χωράν ένα φορτηγό, στους δρόμους μπορούν να περάσουν δύο ταυτόχρονα.

| Προϋποθέσεις | Αποτελέσματα |
|--------------|---------------|
| truck(?t1), truck(?t2), platform(?p1), platform(?p2) | <span style="background-color:#90EE90;color:black">on(?t1, ?p2)</span>, <span style="background-color:#90EE90;color:black">on(?t2, ?p1)</span> |
| on(?t1, ?p1), on(?t2, ?p2) | <span style="background-color:#f8d7da;color:black">not on(?t1, ?p1)</span>, <span style="background-color:#f8d7da;color:black">not on(?t2, ?p2)</span> |
| adj(?p1, ?p2) ∨ adj(?p2, ?p1) | |

### load_container_from_platform (?t ?c ?p)

Το load_container_from_platform είναι μία ενέργεια που φορτώνει ένα container από μία πλατφόρμα σε ένα φορτηγό.
Βασική προϋπόθεση είναι το container να είναι ελεύθερo και το φορτηγό να είναι ελεύθερο. Επίσης το container δεν πρέπει να έχει container από κάτω του. Το αποτέλεσμα είναι ότι το container δεν είναι πλέον ελεύθερο αλλά βρίσκεται πάνω στο φορτηγό και το φορτηγό δεν είναι ελεύθερο.

| Προϋποθέσεις | Αποτελέσματα |
|--------------|---------------|
| truck(?t), container(?c), platform(?p) | <span style="background-color:#90EE90;color:black">on(?c, ?t)</span> |
| free(?t), free(?c), on(?t, ?p), on(?c, ?p) | <span style="background-color:#f8d7da;color:black">not on(?c, ?p)</span>, <span style="background-color:#f8d7da;color:black">not free(?c)</span>, <span style="background-color:#f8d7da;color:black">not free(?t)</span> |
| not has_container_below(?c) | |

### load_container_from_container (?t ?c_above ?c_below ?p)

Αυτή η ενέργεια δημιουργήθηκε στην περίπτωση που τα container είναι στοιβαγμένα. Αυτή η ενέργεια κρίθηκε απαραίτητη καθώς έχει επίδραση και στο container από κάτω.

| Προϋποθέσεις | Αποτελέσματα |
|--------------|---------------|
| truck(?t), container(?c_above), container(?c_below), platform(?p) | <span style="background-color:#90EE90;color:black">on(?c_above, ?t)</span>, <span style="background-color:#90EE90;color:black">free(?c_below)</span> |
| on(?c_above, ?c_below), on(?c_above, ?p), on(?c_below, ?p) | <span style="background-color:#f8d7da;color:black">not on(?c_above, ?c_below)</span>, <span style="background-color:#f8d7da;color:black">not has_container_below(?c_above)</span> |
| free(?t), free(?c_above), on(?t, ?p), has_container_below(?c_above) | <span style="background-color:#f8d7da;color:black">not free(?c_above)</span>, <span style="background-color:#f8d7da;color:black">not on(?c_above, ?p)</span> |

### unload_container_to_platform (?t ?c ?p)

Ξεφορτώνει ένα container από ένα φορτηγό σε μία πλατφόρμα. Βασική προϋπόθεση είναι το φορτηγό να έχει από πάνω του ένα container.

| Προϋποθέσεις | Αποτελέσματα |
|--------------|---------------|
| truck(?t), container(?c), platform(?p) | <span style="background-color:#90EE90;color:black">free(?t)</span>, <span style="background-color:#90EE90;color:black">on(?c, ?p)</span>, <span style="background-color:#90EE90;color:black">free(?c)</span> |
| on(?c, ?t), on(?t, ?p) | <span style="background-color:#f8d7da;color:black">not on(?c, ?t)</span> |

### unload_container_to_container (?t ?p ?c_s ?c_t)

Αυτή η συνάρτηση χρησιμοποιήθηκε για την στοίβαξη των container.

| Προϋποθέσεις | Αποτελέσματα |
|--------------|---------------|
| truck(?t), platform(?p), container(?c_s), container(?c_t) | <span style="background-color:#90EE90;color:black">on(?c_s, ?c_t)</span>, <span style="background-color:#90EE90;color:black">on(?c_s, ?p)</span>, <span style="background-color:#90EE90;color:black">has_container_below(?c_s)</span> |
| on(?t, ?p), on(?c_t, ?p), free(?c_t), on(?c_s, ?t) | <span style="background-color:#90EE90;color:black">free(?t)</span>, <span style="background-color:#90EE90;color:black">free(?c_s)</span>, <span style="background-color:#f8d7da;color:black">not on(?c_s, ?t)</span>, <span style="background-color:#f8d7da;color:black">not free(?c_t)</span> |

## Περιγραφή του προβλήματος

### Αρχική κατάσταση
- **Προβλήτες:** p1, p2, p3 (γειτονικές: p1-p2, p1-p3)
- **Containers:** c1, c2, c3
- **Φορτηγό:** t1 στην p2
- **Στοίβες:** Στην p1 στοίβα (c1←c3), στην p2 μόνο το c2

### Κατάσταση στόχου
- Το φορτηγό t1 στην p1
- Στην p3 δύο στοίβες: c3 μόνο του, και c1 πάνω στο c2 (c2←c1)
- Ελεύθερες: p2, p3, c1, c3

## Στατιστικά επίλυσης

**Αριθμός βημάτων πλάνου:** 17

### Γράφος βημάτων πλάνου

1. (move_truck t1 p2 p1)
2. (load_container_from_container t1 c3 c1 p1)
3. (unload_container_to_platform t1 c3 p1)
4. (move_truck t1 p1 p2)
5. (load_container_from_platform t1 c2 p2)
6. (move_truck t1 p2 p1)
7. (move_truck t1 p1 p3)
8. (unload_container_to_platform t1 c2 p3)
9. (move_truck t1 p3 p1)
10. (load_container_from_platform t1 c1 p1)
11. (move_truck t1 p1 p3)
12. (unload_container_to_container t1 p3 c1 c2)
13. (move_truck t1 p3 p1)
14. (load_container_from_platform t1 c3 p1)
15. (move_truck t1 p1 p3)
16. (unload_container_to_platform t1 c3 p3)
17. (move_truck t1 p3 p1)
