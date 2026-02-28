"""
Canadian Geography and Statistics
Contains information about Canadian provinces, cities, and landmarks
"""

class CanadianProvinces:
    """Data about Canadian provinces and territories"""
    
    provinces = {
        "Ontario": {"capital": "Toronto", "population": "14.7M"},
        "Quebec": {"capital": "Quebec City", "population": "8.5M"},
        "British Columbia": {"capital": "Victoria", "population": "5.4M"},
        "Alberta": {"capital": "Edmonton", "population": "4.7M"},
        "Manitoba": {"capital": "Winnipeg", "population": "1.4M"},
        "Saskatchewan": {"capital": "Regina", "population": "1.2M"},
        "Nova Scotia": {"capital": "Halifax", "population": "1.0M"},
        "New Brunswick": {"capital": "Fredericton", "population": "0.8M"},
        "Newfoundland and Labrador": {"capital": "St. John's", "population": "0.5M"},
        "Prince Edward Island": {"capital": "Charlottetown", "population": "0.2M"},
    }
    
    def get_province_info(self, province_name):
        """Retrieve information about a specific province"""
        return self.provinces.get(province_name, "Province not found")


class CanadianCities:
    """Data about major Canadian cities"""
    
    major_cities = {
        "Toronto": {"province": "Ontario", "population": "2.9M", "landmark": "CN Tower"},
        "Vancouver": {"province": "British Columbia", "population": "0.6M", "landmark": "Stanley Park"},
        "Montreal": {"province": "Quebec", "population": "1.7M", "landmark": "Notre-Dame Basilica"},
        "Calgary": {"province": "Alberta", "population": "1.3M", "landmark": "Calgary Tower"},
        "Ottawa": {"province": "Ontario", "population": "1.0M", "landmark": "Parliament Hill"},
        "Winnipeg": {"province": "Manitoba", "population": "0.8M", "landmark": "The Forks"},
        "Quebec City": {"province": "Quebec", "population": "0.5M", "landmark": "Montmorency Falls"},
        "Halifax": {"province": "Nova Scotia", "population": "0.4M", "landmark": "Peggy's Cove"},
    }
    
    def get_city_info(self, city_name):
        """Retrieve information about a specific city"""
        return self.major_cities.get(city_name, "City not found")
    
    def list_all_cities(self):
        """Return list of all major cities"""
        return list(self.major_cities.keys())
