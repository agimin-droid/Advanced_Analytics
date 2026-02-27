/**
 * pages.config.js - Page routing configuration
 * 
 * This file is AUTO-GENERATED. Do not add imports or modify PAGES manually.
 * Pages are auto-registered when you create files in the ./pages/ folder.
 * 
 * THE ONLY EDITABLE VALUE: mainPage
 * This controls which page is the landing page (shown when users visit the app).
 * 
 * Example file structure:
 * 
 *   import HomePage from './pages/HomePage';
 *   import Dashboard from './pages/Dashboard';
 *   import Settings from './pages/Settings';
 *   
 *   export const PAGES = {
 *       "HomePage": HomePage,
 *       "Dashboard": Dashboard,
 *       "Settings": Settings,
 *   }
 *   
 *   export const pagesConfig = {
 *       mainPage: "HomePage",
 *       Pages: PAGES,
 *   };
 * 
 * Example with Layout (wraps all pages):
 *
 *   import Home from './pages/Home';
 *   import Settings from './pages/Settings';
 *   import __Layout from './Layout.jsx';
 *
 *   export const PAGES = {
 *       "Home": Home,
 *       "Settings": Settings,
 *   }
 *
 *   export const pagesConfig = {
 *       mainPage: "Home",
 *       Pages: PAGES,
 *       Layout: __Layout,
 *   };
 *
 * To change the main page from HomePage to Dashboard, use find_replace:
 *   Old: mainPage: "HomePage",
 *   New: mainPage: "Dashboard",
 *
 * The mainPage value must match a key in the PAGES object exactly.
 */
import Admin from './pages/Admin';
import Bivariate from './pages/Bivariate';
import Dashboard from './pages/Dashboard';
import DataHandling from './pages/DataHandling';
import Explorer from './pages/Explorer';
import Insights from './pages/Insights';
import MLRDoE from './pages/MLRDoE';
import PCA from './pages/PCA';
import Preprocessing from './pages/Preprocessing';
import QualityControl from './pages/QualityControl';
import Univariate from './pages/Univariate';
import Upload from './pages/Upload';
import Visualize from './pages/Visualize';
import TTest from './pages/TTest';
import __Layout from './Layout.jsx';


export const PAGES = {
    "Admin": Admin,
    "Bivariate": Bivariate,
    "Dashboard": Dashboard,
    "DataHandling": DataHandling,
    "Explorer": Explorer,
    "Insights": Insights,
    "MLRDoE": MLRDoE,
    "PCA": PCA,
    "Preprocessing": Preprocessing,
    "QualityControl": QualityControl,
    "Univariate": Univariate,
    "Upload": Upload,
    "Visualize": Visualize,
    "TTest": TTest,
}

export const pagesConfig = {
    mainPage: "Dashboard",
    Pages: PAGES,
    Layout: __Layout,
};