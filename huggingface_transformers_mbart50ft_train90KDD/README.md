<h3>Mbart50 fine-tuning experiments using Huggingface Transformers</h3>

<b>Training set</b>
<br/>parallel-27.04.2021-train90KDD.si-en-ta.en 
<br/>parallel-27.04.2021-train90KDD.si-en-ta.si
<br/>parallel-27.04.2021-train90KDD.si-en-ta.ta

<b>Tuning set</b>
<br/>parallel-27.04.2021-tu.un.si-en-ta.en
<br/>parallel-27.04.2021-tu.un.si-en-ta.si
<br/>parallel-27.04.2021-tu.un.si-en-ta.ta

<b>Testing set</b>
<br/>parallel-27.04.2021-ts.un.si-en-ta.en
<br/>parallel-27.04.2021-ts.un.si-en-ta.si
<br/>parallel-27.04.2021-ts.un.si-en-ta.ta


<table>
  <tr>
    <th>Direction</th>
    <th>Checkpoint</th>
    <th>Val Score</th>
    <th>Test Score</th>
  </tr>
  <tr>
    <td>Si->En</td>
    <td>run11-checkpoint-335000</td>
    <td>39.3</td>
    <td>38.1</td>
  </tr>
  <tr>
    <td>En->Si</td>
    <td>run2-checkpoint-450000</td>
    <td>36.0</td>
    <td>34.5</td>
  </tr>
  <tr>
    <td>Si->Ta</td>
    <td>run1-checkpoint-370000</td>
    <td>24.4</td>
    <td>23.1</td>
  </tr>
  <tr>
    <td>Ta->Si</td>
    <td>run1-checkpoint-370000</td>
    <td>33.8</td>
    <td>33.5</td>
  </tr>
  <tr>
    <td>Ta->En</td>
    <td>run1-checkpoint-370000</td>
    <td>37.3</td>
    <td>36.0</td>
  </tr>
  <tr>
    <td>En->Ta</td>
    <td>run4-checkpoint-360000</td>
    <td>23.8</td>
    <td>21.7</td>
  </tr>
</table>





