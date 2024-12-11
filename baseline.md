**Описание построения бейзлайна**

Процесс построения бейзлайна состоит из нескольких частей:
1. Feature Engineering (сбор показателей из сырого датасета для обучения модели)
2. Построение линейной модели (логистическая регрессия)
3. ...
4. Выбор метрики


 *Feature Engineering*
 
Для того, чтобы собрать фичи, мы ознакомились с опытом других исследователей. В качестве базового feature engineering для линейной модели мы изучили пример из статьи https://habr.com/ru/articles/456226/.

*Процесс сбора*:
- Для каждой команды рассчитываются средние показатели за последние 10 матчей (по всем переменным)
- Для каждого матча рассчитывается разница между вектором статистик первой и второй команд. Вектор результата записывается относительно первой команды: 1 - победа первой команды, 0 - ничья или проигрыш.
- Вектор разницы используется для обучения логистической регрессии.

В текущем сетапе как фичи используются все доступные переменные. Изначально использовались только вещественные, но добавление категории venue дополнителным столбцом в вектор улучшило качество предсказаний.

*Logistic Regression*

Для бейзлайна была использована логистическая регрессия, так как помимо определения исхода матча важно видеть, какова вероятность исхода. Эти данные в дальнейшем могут использоваться для составлений предположений о ничьей, проигрыше с большим/меньшим отрывом.
Изменение типа регрессии (с l2 на l1, solver на liblinear) не дало принципиально новых результатов.
Логистическая регрессия выдает вероятности победы. В отдельных случаях разница между вероятностью победы и проигрыша достаточно невысокая (например, 0,4 и 0,6), в других почти однозначна победа команды. Используя эти вероятности можно рассчитать бизнес-метрику для определения окупаемости ставок.
Коэффициент может быть высчитан как 1/P(вероятность победы), а дальшне для оценки прибыльности вложения можно рассчитать ожидаемый ROI = (1/p-1) - (1-p)

Для обучения логистической регрессии изначальный датасет был преобразован в таблицу с фичами (для каждого матча статистики команд за 10 прошлых матчей). Модель была обучена на данном датасете.
На тесте результат выбранной метрики F1 показал 0,6 - что и будет отправной точкой для улучшения модели.
