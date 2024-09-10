import os
import unittest
from time import time
from main import main  # Importiere die main() Funktion aus deinem Spiel

class MainTestCase(unittest.TestCase):
    def test_play(self, agent, scenario, rounds):
        start_time = time()

        main([
            "play",
            "--agents", agent,
            "--scenario", scenario,
            "--n-rounds", str(rounds),
            "--no-gui"
        ])

        # Überprüfe, ob das Logfile existiert
        self.assertTrue(os.path.isfile("logs/game.log"))

        # Überprüfe, ob das Logfile nach dem Start geschrieben wurde
        self.assertGreater(os.path.getmtime("logs/game.log"), start_time)

        # Extrahiere Statistiken aus dem Logfile
        stats = self.extract_stats_from_log("logs/game.log")
        
        # Berechne Durchschnittswerte
        avg_coins_per_round = stats['coins_collected'] / rounds
        avg_steps_per_round = stats['steps_survived'] / rounds

        # Gib die Statistiken aus, die du für deinen Bericht brauchst
        print(f"Agenten-Performance für {agent} im Szenario {scenario}:")
        print(f"Anzahl der Runden: {rounds}")
        print(f"Schritte überlebt: {stats['steps_survived']}")
        print(f"Münzen eingesammelt: {stats['coins_collected']}")
        print(f"Boxen zerstört: {stats['boxes_destroyed']}")
        print(f"Gegner getötet: {stats['enemies_killed']}")
        print(f"Durchschnittliche Schritte pro Runde: {avg_steps_per_round:.2f}")
        print(f"Durchschnittliche Münzen pro Runde: {avg_coins_per_round:.2f}")

    def extract_stats_from_log(self, log_file_path):
        """
        Extrahiert Statistiken wie Schritte, Münzen, zerstörte Boxen und getötete Gegner aus dem Logfile.
        """
        stats = {
            "steps_survived": 0,
            "coins_collected": 0,
            "boxes_destroyed": 0,
            "enemies_killed": 0
        }
        
        # Lies das Logfile und extrahiere relevante Informationen
        with open(log_file_path, 'r') as log_file:
            lines = log_file.readlines()
            
        for line in lines:
            if "picked up coin" in line:
                stats["coins_collected"] += 1
            elif "STARTING STEP" in line:
                stats["steps_survived"] += 1
        
        return stats

if __name__ == '__main__':
    # Hier kannst du den Agenten, das Szenario und die Anzahl der Runden direkt übergeben
    agent_name = "lisa_coin_sym_agent"  # Ersetze das durch den gewünschten Agentennamen
    scenario_name = "coin-heaven"   # Ersetze das durch das gewünschte Szenario
    number_of_rounds = 100          # Ersetze das durch die gewünschte Rundenanzahl

    # Test mit den angegebenen Parametern starten
    test_case = MainTestCase()
    test_case.test_play(agent=agent_name, scenario=scenario_name, rounds=number_of_rounds)
