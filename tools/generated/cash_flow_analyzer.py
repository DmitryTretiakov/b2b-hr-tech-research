import math

def cash_flow_analyzer(cash_flows: list[float], discount_rate: float) -> dict:
    """
    Анализирует ряд денежных потоков для расчета ключевых финансовых показателей.

    Этот инструмент вычисляет чистую приведенную стоимость (NPV), внутреннюю норму доходности (IRR)
    и простой срок окупаемости на основе предоставленного списка денежных потоков и ставки дисконтирования.
    Все расчеты производятся с использованием стандартных библиотек Python без внешних зависимостей
    вроде numpy или scipy.

    Args:
        cash_flows (list[float]): Список денежных потоков. Первый элемент (с индексом 0)
            представляет собой первоначальные инвестиции и должен быть отрицательным числом.
            Последующие элементы представляют собой денежные потоки за периоды 1, 2, 3 и т.д.
        discount_rate (float): Ставка дисконтирования, используемая для расчета NPV.
            Выражается в виде десятичной дроби (например, 10% = 0.1).
            Должна быть больше -1.0.

    Returns:
        dict: Словарь, содержащий результаты анализа.
              В случае успеха:
              {
                  "status": "success",
                  "data": {
                      "npv": float,  // Чистая приведенная стоимость
                      "irr": float,  // Внутренняя норма доходности (может быть строкой с ошибкой)
                      "payback_period_years": float // Срок окупаемости в годах (может быть inf)
                  }
              }

    Raises:
        Exception: Вызывается в следующих случаях:
            - Если список `cash_flows` пуст.
            - Если `discount_rate` меньше или равна -1.0.
            - Если не удается вычислить IRR (например, из-за отсутствия сходимости).
            - При любой другой непредвиденной ошибке во время вычислений.
    """
    # --- Начало основной логики в блоке try-except ---
    try:
        # --- 1. Валидация входных данных ---
        if not isinstance(cash_flows, list) or not cash_flows:
            raise ValueError("Список денежных потоков 'cash_flows' не может быть пустым.")
        if not all(isinstance(cf, (int, float)) for cf in cash_flows):
            raise ValueError("Все элементы в 'cash_flows' должны быть числами.")
        if not isinstance(discount_rate, (int, float)):
            raise ValueError("Ставка дисконтирования 'discount_rate' должна быть числом.")
        if discount_rate <= -1.0:
            raise ValueError("Ставка дисконтирования 'discount_rate' должна быть больше -1.0.")

        # --- 2. Вложенные функции для расчетов (инкапсуляция логики) ---

        def _calculate_npv(rate: float, flows: list[float]) -> float:
            """Вспомогательная функция для расчета NPV."""
            if rate <= -1.0:
                # Технически проверено выше, но для надежности функции
                raise ValueError("Ставка для NPV должна быть > -1.0")
            
            total_npv = 0.0
            for t, flow in enumerate(flows):
                total_npv += flow / ((1 + rate) ** t)
            return total_npv

        def _calculate_irr(flows: list[float], max_iter: int = 1000, tolerance: float = 1e-6) -> float:
            """
            Вспомогательная функция для расчета IRR методом секущих.
            Этот метод итеративно ищет ставку, при которой NPV = 0.
            """
            # Для метода секущих нужны две начальные догадки.
            # Выбираем их так, чтобы они с высокой вероятностью "обхватывали" корень.
            rate1, rate2 = 0.1, 0.11 

            npv1 = _calculate_npv(rate1, flows)

            for _ in range(max_iter):
                npv2 = _calculate_npv(rate2, flows)

                if abs(npv2) < tolerance:
                    return rate2

                # Формула метода секущих для нахождения следующего приближения
                denominator = npv2 - npv1
                if abs(denominator) < 1e-12:
                    # Если NPV не меняется, метод застревает. Выход с ошибкой.
                    raise RuntimeError("Не удалось найти IRR: NPV не чувствителен к изменению ставки (деление на ноль).")

                rate_next = rate2 - npv2 * (rate2 - rate1) / denominator

                rate1, npv1 = rate2, npv2
                rate2 = rate_next

            raise RuntimeError(f"Не удалось вычислить IRR за {max_iter} итераций. Метод не сошелся.")

        def _calculate_payback_period(flows: list[float]) -> float:
            """Вспомогательная функция для расчета простого срока окупаемости."""
            if not flows or flows[0] >= 0:
                # Если нет первоначальных инвестиций, окупаемость равна 0.
                return 0.0

            initial_investment = abs(flows[0])
            cumulative_cash_flow = 0.0
            
            for period, flow in enumerate(flows[1:], start=1):
                if flow <= 0: # Пропускаем периоды с отрицательным или нулевым потоком
                    continue
                
                cumulative_flow_before = cumulative_cash_flow
                cumulative_cash_flow += flow

                if cumulative_cash_flow >= initial_investment:
                    # Окупаемость наступила в этом периоде.
                    # Рассчитываем дробную часть года.
                    amount_needed = initial_investment - cumulative_flow_before
                    fractional_period = amount_needed / flow
                    return (period - 1) + fractional_period

            # Если цикл завершился, а инвестиции не окупились.
            return float('inf')

        # --- 3. Выполнение расчетов ---
        
        npv_value = _calculate_npv(discount_rate, cash_flows)
        payback_period_value = _calculate_payback_period(cash_flows)
        
        irr_value = None
        try:
            # Расчет IRR может не сойтись, обрабатываем это отдельно,
            # чтобы не прерывать весь анализ.
            irr_value = _calculate_irr(cash_flows)
        except RuntimeError as e:
            irr_value = f"Ошибка вычисления: {e}"

        # --- 4. Формирование и возврат результата ---
        return {
            "status": "success",
            "data": {
                "npv": npv_value,
                "irr": irr_value,
                "payback_period_years": payback_period_value,
            }
        }

    except ValueError as e:
        # Перехват ошибок валидации
        raise Exception(f"Ошибка входных данных: {e}")
    except Exception as e:
        # Перехват всех остальных непредвиденных ошибок
        raise Exception(f"Произошла непредвиденная ошибка при анализе денежных потоков: {e}")